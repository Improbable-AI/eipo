
import torch
from torch import nn
import numpy as np
import cv2

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.models.curiosity.encoders import *
from rlpyt.models.curiosity.forward_models import *

class Disagreement(nn.Module):
    """Curiosity model for intrinsically motivated agents: similar to ICM
    except there is an ensemble of forward models that each make predictions.
    The intrinsic reward is defined as the variance between these predictions.
    """

    def __init__(
            self,
            image_shape,
            action_size,
            ensemble_size=5,
            feature_encoding='idf',
            batch_norm=False,
            prediction_beta=1.0,
            obs_stats=None,
            device="cpu",
            forward_loss_wt=0.2,
            forward_model='res', **kwargs
            ):
        super(Disagreement, self).__init__()

        self.ensemble_size = ensemble_size
        self.beta = prediction_beta
        self.feature_encoding = feature_encoding
        self.obs_stats = obs_stats
        self.device = torch.device("cuda:0" if device == "gpu" else "cpu")

        if self.obs_stats is not None:
            self.obs_mean, self.obs_std = self.obs_stats

        if forward_loss_wt == -1.0:
            self.forward_loss_wt = 1.0
        else:
            self.forward_loss_wt = forward_loss_wt

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

        if forward_model == 'res':
            fmodel_class = ResForward
        elif forward_model == 'og':
            fmodel_class = OgForward

        self.forward_model_1 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_2 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_3 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_4 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_5 = fmodel_class(feature_size=self.feature_size, action_size=action_size).to(self.device)

    def forward(self, obs1, obs2, action):

        if self.obs_stats is not None:
            img1 = (obs1 - self.obs_mean) / (self.obs_std+1e-10)
            img2 = (obs2 - self.obs_mean) / (self.obs_std+1e-10)

        # img = np.squeeze(obs1.data.numpy()[20][0])
        # cv2.imwrite('disimages/original.png', img.transpose(1, 2, 0))

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
            phi1 = phi1.view(T, B, -1) # make sure you're not mixing data up here
            phi2 = phi2.view(T, B, -1)

        predicted_phi2 = []

        predicted_phi2.append(self.forward_model_1(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_2(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_3(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_4(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_5(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2_stacked = torch.stack(predicted_phi2)

        return phi2, predicted_phi2, predicted_phi2_stacked

    def compute_bonus(self, observations, next_observations, actions):
        _, _, predicted_phi2_stacked = self.forward(observations, next_observations, actions)
        feature_var = torch.var(predicted_phi2_stacked, dim=0) # feature variance across forward models
        reward = torch.mean(feature_var, axis=-1) # mean over feature
        return self.beta * reward

    def compute_loss(self, observations, next_observations, actions, valid):
        #------------------------------------------------------------#
        # hacky dimension add for when you have only one environment (debugging)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        #------------------------------------------------------------#
        phi2, predicted_phi2, _ = self.forward(observations, next_observations, actions)

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

        return self.forward_loss_wt*forward_loss






