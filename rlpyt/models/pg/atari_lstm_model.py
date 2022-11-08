
import numpy as np
import torch
import torch.nn.functional as F

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel

from rlpyt.models.curiosity.encoders import UniverseHead, BurdaHead, MazeHead
from rlpyt.models.curiosity.disagreement import Disagreement
from rlpyt.models.curiosity.icm import ICM
from rlpyt.models.curiosity.micm import MICM
from rlpyt.models.curiosity.ndigo import NDIGO
from rlpyt.models.curiosity.rnd import RND

RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work


class AtariLstmModel(torch.nn.Module):
    """Recurrent model for Atari agents: a convolutional network into an FC layer
    into an LSTM which outputs action probabilities and state-value estimate.
    """

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,  # Between conv and lstm.
            lstm_size=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            curiosity_kwargs=dict(curiosity_alg='none'),
            obs_stats=None
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()

        self.obs_stats = obs_stats
        if self.obs_stats is not None:
            self.obs_mean, self.obs_std = self.obs_stats

        self.dual_value = curiosity_kwargs.get('dual_value', False)
        self.dual_policy = curiosity_kwargs.get('dual_policy', 'default')
        self.use_minmax = curiosity_kwargs.get('use_minmax', False)
        if curiosity_kwargs['curiosity_alg'] != 'none':
            curiosity_init_kwargs = {k: curiosity_kwargs[k] for k in curiosity_kwargs.keys() - {'curiosity_alg', 'dual_value', 'dual_policy'}}
            if curiosity_kwargs['curiosity_alg'] == 'icm':
                self.curiosity_model = ICM(image_shape=image_shape, action_size=output_size, **curiosity_init_kwargs)
            if curiosity_kwargs['curiosity_alg'] == 'micm':
                self.curiosity_model = MICM(image_shape=image_shape, action_size=output_size, **curiosity_init_kwargs)
            elif curiosity_kwargs['curiosity_alg'] == 'disagreement':
                self.curiosity_model = Disagreement(image_shape=image_shape, action_size=output_size, **curiosity_init_kwargs)
            elif curiosity_kwargs['curiosity_alg'] == 'ndigo':
                self.curiosity_model = NDIGO(image_shape=image_shape, action_size=output_size, obs_stats=self.obs_stats, **curiosity_init_kwargs)
            elif curiosity_kwargs['curiosity_alg'] == 'rnd':
                self.curiosity_model = RND(image_shape=image_shape, obs_stats=self.obs_stats, **curiosity_init_kwargs)

            if curiosity_kwargs['feature_encoding'] == 'idf':
                self.conv = UniverseHead(image_shape=image_shape,
                                         batch_norm=curiosity_kwargs['batch_norm'])
                self.conv.output_size = self.curiosity_model.feature_size
            elif curiosity_kwargs['feature_encoding'] == 'idf_burda':
                self.conv = BurdaHead(image_shape=image_shape,
                                      output_size=self.curiosity_model.feature_size,
                                      batch_norm=curiosity_kwargs['batch_norm'])
                self.conv.output_size = self.curiosity_model.feature_size
            elif curiosity_kwargs['feature_encoding'] == 'idf_maze':
                self.conv = MazeHead(image_shape=image_shape,
                                     output_size=self.curiosity_model.feature_size,
                                     batch_norm=curiosity_kwargs['batch_norm'])
                self.conv.output_size = self.curiosity_model.feature_size
            elif curiosity_kwargs['feature_encoding'] == 'none':
                self.conv = Conv2dHeadModel(image_shape=image_shape,
                                            channels=channels or [16, 32],
                                            kernel_sizes=kernel_sizes or [8, 4],
                                            strides=strides or [4, 2],
                                            paddings=paddings or [0, 1],
                                            use_maxpool=use_maxpool,
                                            hidden_sizes=fc_sizes) # Applies nonlinearity at end.

        else:
            self.conv = Conv2dHeadModel(
                image_shape=image_shape,
                channels=channels or [16, 32],
                kernel_sizes=kernel_sizes or [8, 4],
                strides=strides or [4, 2],
                paddings=paddings or [0, 1],
                use_maxpool=use_maxpool,
                hidden_sizes=fc_sizes, # Applies nonlinearity at end.
            )

        self.lstm = torch.nn.LSTM(self.conv.output_size + output_size, lstm_size)
        self.pi = torch.nn.Linear(lstm_size, output_size)

        if self.use_minmax:
          self.pi_ext = torch.nn.Linear(lstm_size, output_size)
        else:
          if self.dual_policy in {'combined', 'int', 'ext'}:
              self.pi_ext = torch.nn.Linear(lstm_size, output_size)
              self.pi_int = torch.nn.Linear(lstm_size, output_size)

        self.value = torch.nn.Linear(lstm_size, 1)
        if self.dual_value:
            self.value_int = torch.nn.Linear(lstm_size, 1)
        if self.use_minmax:
          # V^{\pi^\prime}_E
          self.value_ext_prime = torch.nn.Linear(lstm_size, 1)

    def forward(self, image, prev_action, prev_reward, init_rnn_state):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        *image_shape], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Recurrent layers processed as [T,B,H]. Used in
        both sampler and in algorithm (both via the agent).  Also returns the
        next RNN state.
        """
        if self.obs_stats is not None: # don't normalize observation
            image = (image - self.obs_mean) / (self.obs_std+1e-10)
        img = image.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        lstm_input = torch.cat([
            fc_out.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)

        pi = F.softmax(self.pi(lstm_out.view(T * B, -1)), dim=-1)
        v = self.value(lstm_out.view(T * B, -1)).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B) # restore leading dimensions: [T,B], [B], or [], as input.
        if self.use_minmax:
          pi_ext = F.softmax(self.pi_ext(lstm_out.view(T * B, -1)), dim=-1)
          pi_ext = restore_leading_dims((pi_ext), lead_dim, T, B)
          pi_int = torch.zeros_like(pi_ext)
        else:
          if self.dual_policy in {'combined', 'int', 'ext'}:
              pi_ext = F.softmax(self.pi_ext(lstm_out.view(T * B, -1)), dim=-1)
              pi_int = F.softmax(self.pi_int(lstm_out.view(T * B, -1)), dim=-1)
              pi_ext = restore_leading_dims((pi_ext), lead_dim, T, B)
              pi_int = restore_leading_dims((pi_int), lead_dim, T, B)
          else:
              pi_ext = torch.tensor([0.0])
              pi_int = torch.tensor([0.0])

        if self.dual_value:
            v_int = self.value_int(lstm_out.view(T * B, -1)).squeeze(-1)
            v_int = restore_leading_dims (v_int, lead_dim, T, B)
        else:
            v_int = torch.tensor([0.0])

        if self.use_minmax:
          v_ext_prime = self.value_ext_prime(lstm_out.view(T * B, -1)).squeeze(-1)
          v_ext_prime = restore_leading_dims (v_ext_prime, lead_dim, T, B)
        else:
          v_ext_prime = torch.tensor([0.0])

        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return pi, pi_ext, pi_int, v, v_int, v_ext_prime, next_rnn_state




