
import torch
from torch import nn

from rlpyt.models.utils import Flatten

class UniverseHead(nn.Module):
    '''
    Universe agent example: https://github.com/openai/universe-starter-agent
    '''
    def __init__(
            self, 
            image_shape,
            batch_norm=False
            ):
        super(UniverseHead, self).__init__()
        c, h, w = image_shape
        sequence = list()
        for l in range(5):
            if l == 0:
                conv = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            else:
                conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            block = [conv, nn.ELU()]
            if batch_norm:
                block.append(nn.BatchNorm2d(32))
            sequence.extend(block)
        self.model = nn.Sequential(*sequence)


    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state.view(encoded_state.shape[0], -1)

class MazeHead(nn.Module):
    '''
    World discovery models paper
    '''
    def __init__(
            self, 
            image_shape,
            output_size=256,
            conv_output_size=256,
            batch_norm=False,
            ):
        super(MazeHead, self).__init__()
        c, h, w = image_shape
        self.output_size = output_size
        self.conv_output_size = conv_output_size
        self.model = nn.Sequential(
                                nn.Conv2d(in_channels=c, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                                nn.ReLU(),
                                Flatten(),
                                nn.Linear(in_features=self.conv_output_size, out_features=self.output_size),
                                nn.ReLU()
                                )

    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state

class BurdaHead(nn.Module):
    '''
    Large scale curiosity paper
    '''
    def __init__(
            self, 
            image_shape,
            output_size=512,
            conv_output_size=3136,
            batch_norm=False,
            ):
        super(BurdaHead, self).__init__()
        c, h, w = image_shape
        self.output_size = output_size
        self.conv_output_size = conv_output_size
        sequence = list()
        sequence += [nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(8, 8), stride=(4, 4)), 
                     nn.LeakyReLU()]
        if batch_norm:
            sequence.append(nn.BatchNorm2d(32))
        sequence += [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)), 
                     nn.LeakyReLU()]
        if batch_norm:
            sequence.append(nn.BatchNorm2d(64))
        sequence += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
                     nn.LeakyReLU()]
        if batch_norm:
            sequence.append(nn.BatchNorm2d(64))
        sequence.append(Flatten())
        sequence.append(nn.Linear(in_features=self.conv_output_size, out_features=self.output_size))

        self.model = nn.Sequential(*sequence)
        # self.model = nn.Sequential(
        #                         nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
        #                         nn.LeakyReLU(),
        #                         # nn.BatchNorm2d(32),
        #                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
        #                         nn.LeakyReLU(),
        #                         # nn.BatchNorm2d(64),
        #                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
        #                         nn.LeakyReLU(),
        #                         # nn.BatchNorm2d(64),
        #                         Flatten(),
        #                         nn.Linear(in_features=self.conv_output_size, out_features=self.output_size),
        #                         # nn.BatchNorm1d(self.output_size)
        #                         )

    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state