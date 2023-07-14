from cmath import sin
from torch import nn
import torch

class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_freqs_xyz=8,
        num_freqs_dir=6,
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        self.num_freqs_xyz = num_freqs_xyz
        self.num_freqs_dir = num_freqs_dir

        self.dim_xyz = 3 + 2 * 3 * num_freqs_xyz
        self.dim_dir = 3 + 2 * 3 * num_freqs_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, xyz, direction):
        xyz = self.positional_encoding(xyz, self.num_freqs_xyz)
        direction = self.positional_encoding(direction, self.num_freqs_dir)
        
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)

    @staticmethod
    def positional_encoding(tensor, num_encoding_functions=6):
        encoding = [tensor]
        
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,    
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        return torch.cat(encoding, dim=-1) 

class MultiHeadNeRFModel(torch.nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = torch.nn.Linear(
            hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 3)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, xyz, direction):
        x = self.positional_encoding(xyz, self.num_encoding_functions)
        direction = self.positional_encoding(direction, self.num_encoding_functions)
        
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        # x = torch.cat((feat, direction), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)
    
    @staticmethod
    def positional_encoding(tensor, num_encoding_functions=6):
        encoding = [tensor]
        
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,    
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        return torch.cat(encoding, dim=-1) 


class NeRF_MLP(nn.Module):
    def __init__(self, freq_num=8):
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

    def forward(self, flat_pos, direction):

        flat_pos = self.positional_encoding(flat_pos, self.freq_num)

        intermediate_rep = self.model_1(flat_pos)

        concat_pos = torch.cat([intermediate_rep, flat_pos], dim=1)

        intermediate_rep = self.model_2(concat_pos)

        density = self.density_head(intermediate_rep)

        rgb = self.rgb_head(intermediate_rep)

        outputs = torch.cat([rgb, density], -1)

        return outputs

    @staticmethod
    def positional_encoding(tensor, num_encoding_functions=6):
        encoding = [tensor]
        
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,    
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        return torch.cat(encoding, dim=-1) 



