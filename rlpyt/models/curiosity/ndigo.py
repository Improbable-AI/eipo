
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

import numpy as np
import torch
from torch import nn
torch.set_printoptions(threshold=500*2*100)

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.graph_utils import save_dot
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead

GruState = namedarraytuple("GruState", ["c"])  # For downstream namedarraytuples to work

class NdigoForward(nn.Module):
    """Frame predictor MLP for NDIGO curiosity algorithm"""
    def __init__(self,
                 feature_size,
                 action_size, # usually multi-action sequence
                 output_size # observation size
                 ):
        super(NdigoForward, self).__init__()

        self.model = nn.Sequential(nn.Linear(feature_size + action_size, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, output_size))

    def forward(self, belief_states, action_seqs):
        predicted_states = self.model(torch.cat([belief_states, action_seqs], 2))
        return predicted_states

class NDIGO(torch.nn.Module):
    """Curiosity model for intrinsically motivated agents: a convolutional network
    into an FC layer into an LSTM into an MLP which outputs forward predictions on
    future states, and computes an intrinsic reward using the error in these predictions.
    """

    def __init__(
            self,
            image_shape,
            action_size,
            horizon,
            prediction_beta=1.0,
            feature_encoding='idf_maze',
            gru_size=128,
            batch_norm=False,
            obs_stats=None,
            device='cpu', **kwargs
            ):
        """Instantiate neural net module according to inputs."""
        super(NDIGO, self).__init__()

        self.action_size = action_size
        self.horizon = horizon
        self.feature_encoding = feature_encoding
        self.obs_stats = obs_stats
        self.beta = prediction_beta
        if self.obs_stats is not None:
            self.obs_mean, self.obs_std = self.obs_stats
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')
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

        self.gru_size = gru_size
        self.gru = torch.nn.GRU(self.feature_size + action_size, self.gru_size)
        self.gru_states = None # state output of last batch - (1, B, gru_size) or None

        self.forward_model_1 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*1,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_2 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*2,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_3 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*3,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_4 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*4,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_5 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*5,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_6 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*6,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_7 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*7,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_8 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*8,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_9 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*9,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_10 = NdigoForward(feature_size=self.gru_size,
                                            action_size=action_size*10,
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])


    def forward(self, observations, prev_actions):

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)

        # encode batch
        images = observations.type(torch.float)
        encoded_states = self.encoder(images.view(T * B, *img_shape)).view(T, B, -1)

        # pass encoded batch through GRU
        # gru_state = None if gru_state is None else gru_state.c
        gru_inputs = torch.cat([encoded_states, prev_actions], dim=2)
        belief_states, gru_output_states = self.gru(gru_inputs, self.gru_states)

        return belief_states, gru_output_states


    def compute_bonus(self, observations, prev_actions, actions):
        #------------------------------------------------------------#
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)

        # hacky dimension add for when you have only one environment
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.view(1, 1, -1)
        if actions.dim() == 1:
            actions = actions.view(1, 1, -1)
        #------------------------------------------------------------#

        # generate belief states
        belief_states, gru_output_states = self.forward(observations, prev_actions)
        self.gru_states = None # only bc we're processing exactly 1 episode per batch

        # slice beliefs and actions
        belief_states_t = belief_states.clone()[:-self.horizon] # slice off last timesteps
        belief_states_tm1 = belief_states.clone()[:-self.horizon-1]

        action_seqs_t = torch.zeros((T-self.horizon, B, self.horizon*self.action_size), device=self.device) # placeholder
        action_seqs_tm1 = torch.zeros((T-self.horizon-1, B, (self.horizon+1)*self.action_size), device=self.device) # placeholder
        for i in range(len(actions)-self.horizon):
            if i != len(actions)-self.horizon-1:
                action_seq_tm1 = actions.clone()[i:i+self.horizon+1]
                action_seq_tm1 = torch.transpose(action_seq_tm1, 0, 1)
                action_seq_tm1 = torch.reshape(action_seq_tm1, (action_seq_tm1.shape[0], -1))
                action_seqs_tm1[i] = action_seq_tm1
            action_seq_t = actions.clone()[i:i+self.horizon]
            action_seq_t = torch.transpose(action_seq_t, 0, 1)
            action_seq_t = torch.reshape(action_seq_t, (action_seq_t.shape[0], -1))
            action_seqs_t[i] = action_seq_t

        # make forward model predictions
        if self.horizon == 1:
            predicted_states_tm1 = self.forward_model_2(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-1, B, 75)
            predicted_states_t = self.forward_model_1(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-1, B, 75)
        elif self.horizon == 2:
            predicted_states_tm1 = self.forward_model_3(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-2, B, 75)
            predicted_states_t = self.forward_model_2(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-2, B, 75)
        elif self.horizon == 3:
            predicted_states_tm1 = self.forward_model_4(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-3, B, 75)
            predicted_states_t = self.forward_model_3(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-3, B, 75)
        elif self.horizon == 4:
            predicted_states_tm1 = self.forward_model_5(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-4, B, 75)
            predicted_states_t = self.forward_model_4(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-4, B, 75)
        elif self.horizon == 5:
            predicted_states_tm1 = self.forward_model_6(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-5, B, 75)
            predicted_states_t = self.forward_model_5(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-5, B, 75)
        elif self.horizon == 6:
            predicted_states_tm1 = self.forward_model_7(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-6, B, 75)
            predicted_states_t = self.forward_model_6(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-6, B, 75)
        elif self.horizon == 7:
            predicted_states_tm1 = self.forward_model_8(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-7, B, 75)
            predicted_states_t = self.forward_model_7(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-7, B, 75)
        elif self.horizon == 8:
            predicted_states_tm1 = self.forward_model_9(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-8, B, 75)
            predicted_states_t = self.forward_model_8(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-8, B, 75)
        elif self.horizon == 9:
            predicted_states_tm1 = self.forward_model_10(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-9, B, 75)
            predicted_states_t = self.forward_model_9(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-9, B, 75)
        elif self.horizon == 10:
            predicted_states_tm1 = self.forward_model_11(belief_states_tm1, action_seqs_tm1.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-10, B, 75)
            predicted_states_t = self.forward_model_10(belief_states_t, action_seqs_t.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-10, B, 75)

        predicted_states_tm1 = nn.functional.sigmoid(predicted_states_tm1)
        true_obs_tm1 = observations.clone()[self.horizon:-1].view(-1, *predicted_states_tm1.shape[1:]).type(torch.float)
        predicted_states_t = nn.functional.sigmoid(predicted_states_t)
        true_obs_t = observations.clone()[self.horizon:].view(-1, *predicted_states_t.shape[1:]).type(torch.float)

        # generate losses
        losses_tm1 = nn.functional.binary_cross_entropy(predicted_states_tm1, true_obs_tm1, reduction='none')
        losses_tm1 = torch.sum(losses_tm1, dim=-1)/losses_tm1.shape[-1] # average of each feature for each environment at each timestep (T, B, ave_loss_over_feature)
        losses_t = nn.functional.binary_cross_entropy(predicted_states_t, true_obs_t, reduction='none')
        losses_t = torch.sum(losses_t, dim=-1)/losses_t.shape[-1]



        # subtract losses to get rewards (r[t+H-1] = losses[t-1] - losses[t])
        r_int = torch.zeros((T, B), device=self.device)
        r_int[self.horizon:len(losses_t)+self.horizon-1] = losses_tm1 - losses_t[1:] # time zero reward is set to 0 (L[-1] doesn't exist)
        # r_int[self.horizon:len(losses_t)+self.horizon-1] = losses_t[1:] - losses_tm1
        # r_int[1:len(losses_t)] = losses_tm1 - losses_t[1:]
        # r_int[1:len(losses_t)] = losses_t[1:] - losses_tm1

        # r_int = nn.functional.relu(r_int)

        return r_int*self.beta


    def compute_loss(self, observations, prev_actions, actions, valid):
        #------------------------------------------------------------#
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)
        # hacky dimension add for when you have only one environment
        if prev_actions.dim() == 2:
            prev_actions = prev_actions.unsqueeze(1)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        #------------------------------------------------------------#

        # generate belief states
        belief_states, gru_output_states = self.forward(observations, prev_actions)
        self.gru_states = None # only bc we're processing exactly 1 episode per batch

        for k in range(1, 11):
            action_seqs = torch.zeros((T-k, B, k*self.action_size), device=self.device) # placeholder
            for i in range(len(actions)-k):
                action_seq = actions[i:i+k]
                action_seq = torch.transpose(action_seq, 0, 1)
                action_seq = torch.reshape(action_seq, (action_seq.shape[0], -1))
                action_seqs[i] = action_seq

            # make forward model predictions for this predictor
            if k == 1:
                predicted_states = self.forward_model_1(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-1, B, 75)
            elif k == 2:
                predicted_states = self.forward_model_2(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-2, B, 75)
            elif k == 3:
                predicted_states = self.forward_model_3(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-3, B, 75)
            elif k == 4:
                predicted_states = self.forward_model_4(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-4, B, 75)
            elif k == 5:
                predicted_states = self.forward_model_5(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-5, B, 75)
            elif k == 6:
                predicted_states = self.forward_model_6(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-6, B, 75)
            elif k == 7:
                predicted_states = self.forward_model_7(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-7, B, 75)
            elif k == 8:
                predicted_states = self.forward_model_8(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-8, B, 75)
            elif k == 9:
                predicted_states = self.forward_model_9(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-9, B, 75)
            elif k == 10:
                predicted_states = self.forward_model_10(belief_states[:-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-10, B, 75)

            # generate losses for this predictor
            predicted_states = nn.functional.sigmoid(predicted_states)
            true_obs = observations[k:].view(-1, *predicted_states.shape[1:]).detach().type(torch.float)

            floss = nn.functional.binary_cross_entropy(predicted_states, true_obs.detach(), reduction='mean')
            if k == 1:
                loss = floss
            else:
                loss += floss

        return loss



