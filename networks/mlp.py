from cmath import sin
from torch import nn
import torch


class NeRF_MLP(nn.Module):
    def __init__(self, freq_num):
        super(NeRF_MLP, self).__init__()
        """
        position_dim: The dimension of the last axis of the points
        """
        self.position_dim = 3 + 3 * 2 * freq_num
        self.freq_num = freq_num

        self.model_1 = nn.Sequential(
            nn.Linear(self.position_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # concatenate with the position vector
        self.model_2 = nn.Sequential(
            nn.Linear(256 + self.position_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # output density value
        self.density_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )

        # output RGB value
        self.rgb_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, flat_pos):

        flat_pos = self.positional_encoding(flat_pos)

        intermediate_rep = self.model_1(flat_pos)

        concat_pos = torch.cat([intermediate_rep, flat_pos], dim=1)

        intermediate_rep = self.model_2(concat_pos)

        density = self.density_head(intermediate_rep)

        rgb = self.rgb_head(intermediate_rep)

        outputs = torch.cat([rgb, density], -1)

        return outputs

    def positional_encoding(self, position):
        terms = [position]
        for i in range(self.freq_num):
            sin_encoding = torch.sin(2 ** i * torch.pi * position)
            cos_encoding = torch.cos(2 ** i * torch.pi * position)
            terms.append(sin_encoding)
            terms.append(cos_encoding)

        return torch.concat(terms, dim=1)