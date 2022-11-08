
import torch
from torch import nn
import numpy as np
import cv2

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.models.curiosity.encoders import *
from rlpyt.models.curiosity.forward_models import *

class ICM(nn.Module):
    """ICM curiosity agent: two neural networks, one
    forward model that predicts the next state, and one inverse model that predicts
    the action given two states. The forward model uses the prediction error to
    compute an intrinsic reward. The inverse model trains features that are invariant
    to distracting environment stochasticity.
    """

    def __init__(
            self,
            image_shape,
            action_size,
            feature_encoding='idf',
            batch_norm=False,
            prediction_beta=1.0,
            obs_stats=None,
            forward_loss_wt=0.2,
            forward_model='res',
            feature_space='inverse',
            encoder_dict=None,
            fix_features=False, **kwargs
            ):
        super(ICM, self).__init__()
        self.beta = prediction_beta
        self.feature_encoding = feature_encoding
        self.feature_space = feature_space
        self.obs_stats = obs_stats
        self.fix_features = fix_features
        if self.obs_stats is not None:
            self.obs_mean, self.obs_std = self.obs_stats

        if forward_loss_wt == -1.0:
            self.forward_loss_wt = 1.0
            self.inverse_loss_wt = 1.0
        else:
            self.forward_loss_wt = forward_loss_wt
            self.inverse_loss_wt = 1-forward_loss_wt

        if self.feature_encoding != 'none':
            if self.feature_encoding == 'idf':
                self.feature_size = 288
                self.encoder = UniverseHead(image_shape=image_shape, batch_norm=batch_norm)
            elif self.feature_encoding == 'idf_burda':
                self.feature_size = 512
                self.encoder = BurdaHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm)
            elif self.feature_encoding == 'idf_maze':
                self.feature_size = 256
                self.encoder = MazeHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm)

            if encoder_dict is not None:
                self.encoder.load_state_dict(encoder_dict)

        if self.feature_space == 'inverse':
            self.inverse_model = nn.Sequential(
                nn.Linear(self.feature_size * 2, self.feature_size),
                nn.ReLU(),
                nn.Linear(self.feature_size, action_size)
                )

        if forward_model == 'res':
            fmodel_class = ResForward
        elif forward_model == 'og':
            fmodel_class = OgForward

        self.forward_model = fmodel_class(feature_size=self.feature_size, action_size=action_size)


    def forward(self, obs1, obs2, action):

        if self.obs_stats is not None:
            img1 = (obs1 - self.obs_mean) / self.obs_std
            img2 = (obs2 - self.obs_mean) / self.obs_std

        # img = np.squeeze(obs1.data.numpy()[20][0])
        # cv2.imwrite('icmimages/original.png', img.transpose(1, 2, 0))

        img1 = obs1.type(torch.float)
        img2 = obs2.type(torch.float) # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs1, 3)

        phi1 = img1
        phi2 = img2
        if self.feature_encoding != 'none':
            phi1 = self.encoder(img1.view(T * B, *img_shape))
            phi2 = self.encoder(img2.view(T * B, *img_shape))
            phi1 = phi1.view(T, B, -1)
            phi2 = phi2.view(T, B, -1)

        predicted_action = None
        if self.feature_space == 'inverse' and self.fix_features == False:
            predicted_action = self.inverse_model(torch.cat([phi1, phi2], 2))
        predicted_phi2 = self.forward_model(phi1.detach(), action.view(T, B, -1).detach())

        return phi1, phi2, predicted_phi2, predicted_action

    def compute_bonus(self, observations, next_observations, actions):
        phi1, phi2, predicted_phi2, _ = self.forward(observations, next_observations, actions)
        reward = nn.functional.mse_loss(predicted_phi2, phi2, reduction='none').sum(-1)/self.feature_size
        return self.beta * reward

    def compute_loss(self, observations, next_observations, actions, valid):
        # dimension add for when you have only one environment
        if actions.dim() == 2: actions = actions.unsqueeze(1)
        phi1, phi2, predicted_phi2, predicted_action = self.forward(observations, next_observations, actions)
        actions = torch.max(actions.view(-1, *actions.shape[2:]), 1)[1] # convert action to (T * B, action_size)
        inverse_loss = torch.tensor(0.0)
        if self.feature_space == 'inverse' and self.fix_features == False:
            inverse_loss = nn.functional.cross_entropy(predicted_action.view(-1, *predicted_action.shape[2:]), actions.detach(), reduction='none').view(phi1.shape[0], phi1.shape[1])
            inverse_loss = valid_mean(inverse_loss, valid.detach())
        forward_loss = nn.functional.mse_loss(predicted_phi2, phi2.detach(), reduction='none').sum(-1)/self.feature_size
        forward_loss = valid_mean(forward_loss, valid.detach())
        return self.inverse_loss_wt*inverse_loss, self.forward_loss_wt*forward_loss





