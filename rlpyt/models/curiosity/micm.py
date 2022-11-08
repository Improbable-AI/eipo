
import numpy as np
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.models.curiosity.encoders import *
from rlpyt.models.curiosity.forward_models import *

class MICM(nn.Module):
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
            ensemble_mode='sample',
            device="cpu", **kwargs
            ):
        super(MICM, self).__init__()
        self.device = torch.device("cuda:0" if device == "gpu" else "cpu")
        self.beta = prediction_beta
        self.feature_encoding = feature_encoding
        self.obs_stats = obs_stats
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

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_size * 2, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, action_size)
            )

        if forward_model == 'res':
            fmodel_class = ResForward
        elif forward_model == 'og':
            fmodel_class = OgForward

        self.ensemble_mode = ensemble_mode

        self.forward_model_1 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_2 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_3 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_4 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_5 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)

    def forward(self, obs1, obs2, action):

        if self.obs_stats is not None:
            img1 = (obs1 - self.obs_mean) / self.obs_std
            img2 = (obs2 - self.obs_mean) / self.obs_std

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

        predicted_action = self.inverse_model(torch.cat([phi1, phi2], 2))

        predicted_phi2 = []
        predicted_phi2.append(self.forward_model_1(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_2(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_3(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_4(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_5(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2_stacked = torch.stack(predicted_phi2)

        return phi1, phi2, predicted_phi2_stacked, predicted_action

    def compute_bonus(self, observations, next_observations, actions):
        phi1, phi2, predicted_phi2, _ = self.forward(observations, next_observations, actions)

        rewards = []
        rewards.append(nn.functional.mse_loss(predicted_phi2[0], phi2, reduction='none').sum(-1)/self.feature_size)
        rewards.append(nn.functional.mse_loss(predicted_phi2[1], phi2, reduction='none').sum(-1)/self.feature_size)
        rewards.append(nn.functional.mse_loss(predicted_phi2[2], phi2, reduction='none').sum(-1)/self.feature_size)
        rewards.append(nn.functional.mse_loss(predicted_phi2[3], phi2, reduction='none').sum(-1)/self.feature_size)
        rewards.append(nn.functional.mse_loss(predicted_phi2[4], phi2, reduction='none').sum(-1)/self.feature_size)
        rewards = torch.stack(rewards)

        if self.ensemble_mode == 'sample':
            reward = rewards[np.random.choice(5)]
        elif self.ensemble_mode == 'mean':
            reward = torch.mean(rewards, dim=0)
        elif self.ensemble_mode == 'var':
            reward = torch.var(rewards, dim=0)

        return self.beta * reward

    def compute_loss(self, observations, next_observations, actions, valid):
        # dimension add for when you have only one environment
        if actions.dim() == 2: actions = actions.unsqueeze(1)
        phi1, phi2, predicted_phi2, predicted_action = self.forward(observations, next_observations, actions)
        actions = torch.max(actions.view(-1, *actions.shape[2:]), 1)[1] # convert action to (T * B, action_size)
        inverse_loss = nn.functional.cross_entropy(predicted_action.view(-1, *predicted_action.shape[2:]), actions.detach(), reduction='none').view(phi1.shape[0], phi1.shape[1])
        inverse_loss = valid_mean(inverse_loss, valid.detach())

        forward_loss = torch.tensor(0.0, device=self.device)

        forward_loss_1 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[0], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_1, valid)

        forward_loss_2 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[1], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_2, valid)

        forward_loss_3 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[2], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_3, valid)

        forward_loss_4 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[3], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_4, valid)

        forward_loss_5 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[4], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_5, valid)

        return self.inverse_loss_wt*inverse_loss, self.forward_loss_wt*forward_loss





