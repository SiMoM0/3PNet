# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made by Simone Mosco @ Department of Information Engineering, Univeristy of Padova, 2025:
# Reorganized code structure, added new features


import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import autocast
import torch.nn.functional as F

class myLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


def build_proj_matrix(
    indices_non_zeros, occupied_cell, batch_size, num_2d_cells, inflate_ind, channels
):
    num_points = indices_non_zeros.shape[1] // batch_size
    matrix_shape = (batch_size, num_2d_cells, num_points)

    cell_ind = inflate_ind.clone()

    # Sparse projection matrix for Inflate step
    inflate = torch.sparse_coo_tensor(
        indices_non_zeros, occupied_cell.reshape(-1), matrix_shape
    ).transpose(1, 2)
    inflate_ind = inflate_ind.unsqueeze(1).expand(-1, channels, -1)
    # clip the indices to the maximum number of cells
    inflate_ind = torch.clamp(inflate_ind, 0, num_2d_cells - 1)

    # Count number of points in each cells (used in flatten step)
    with autocast("cuda", enabled=False):
        # torch < 2.x
        #num_points_per_cells = torch.bmm(
        #    inflate, torch.bmm(inflate.transpose(1, 2), occupied_cell.unsqueeze(-1))
        #)
        # torch >= 2.x
        num_points_per_cells = []
        for i in range(batch_size):
            num_points_per_cells.append(
                torch.matmul(
                    inflate[i],
                    torch.matmul(inflate[i].transpose(0, 1), occupied_cell[i].unsqueeze(-1)),
                )
            )
        num_points_per_cells = torch.stack(num_points_per_cells, dim=0)

    # Sparse projection matrix for Flatten step (projection & average in each 2d cells)
    weight_per_point = 1.0 / (num_points_per_cells.reshape(-1) + 1e-6)
    weight_per_point *= occupied_cell.reshape(-1)
    flatten = torch.sparse_coo_tensor(indices_non_zeros, weight_per_point, matrix_shape)

    return {"flatten": flatten, "inflate": inflate_ind, "cell_ind":cell_ind, "weight":weight_per_point}


class DropPath(nn.Module):
    """
    Stochastic Depth

    Original code of this module is at:
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def extra_repr(self):
        return f"prob={self.drop_prob}"

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output


class ChannelMix(nn.Module):
    def __init__(self, channels, drop_path_prob, layer_norm=False):
        super().__init__()
        self.compressed = False
        self.layer_norm = layer_norm
        if layer_norm:
           self.norm = myLayerNorm(channels)
        else:
           self.norm = nn.BatchNorm1d(channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale
        self.drop_path = DropPath(drop_path_prob)


    def compress(self):

        if self.layer_norm:
            warnings.warn("Compression of ChannelMix layer in WaffleIron has not been implemented with layer norm.")
            return
        # Join Batch norm and first conv
        norm_weight = self.norm.weight.data / torch.sqrt(
            self.norm.running_var.data + 1e-05
        )
        norm_bias = self.norm.bias.data - norm_weight * self.norm.running_mean.data
        

        # Careful the order of the two lines below should not be changed
        self.mlp[0].bias.data = (
            self.mlp[0].weight.data[:, :, 0] @ norm_bias + self.mlp[0].bias.data
        )
        self.mlp[0].weight.data = self.mlp[0].weight.data * norm_weight[None, :, None]
        # Join scale and last conv
        self.mlp[-1].weight.data = self.mlp[-1].weight.data * self.scale.weight.data
        self.mlp[-1].bias.data = (
            self.mlp[-1].bias.data * self.scale.weight.data[:, 0, 0]
        )
        # Flag
        self.compressed = True


    def forward(self, tokens):
        """tokens <- tokens + LayerScale( MLP( BN(tokens) ) )"""

        if self.compressed:
            assert not self.training
            return tokens + self.drop_path(self.mlp(tokens))
        else:
            return tokens + self.drop_path(self.scale(self.mlp(self.norm(tokens))))

class SpatialMix(nn.Module):
    def __init__(self, channels, grid_shape, drop_path_prob, layer_norm=False):
        super().__init__()
        self.compressed = False
        self.H, self.W = grid_shape
        if layer_norm:
            self.norm = myLayerNorm(channels)
        else:
            self.norm = nn.BatchNorm1d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        self.att = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False, groups=channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        )
        self.grid_shape = grid_shape
        self.drop_path = DropPath(drop_path_prob)

    def extra_repr(self):
        return f"(grid): [{self.grid_shape[0]}, {self.grid_shape[1]}]"

    def compress(self):
        # Join scale and last conv
        self.ffn[-1].weight.data = (
            self.ffn[-1].weight.data * self.scale.weight.data[..., None]
        )
        self.ffn[-1].bias.data = (
            self.ffn[-1].bias.data * self.scale.weight.data[:, 0, 0]
        )
        # Flag
        self.compressed = True

    def forward_compressed(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        # Make sure we are not in training mode
        assert not self.training
        # Forward pass
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        # Flatten
        with autocast("cuda", enabled=False):

            residual = residual * sp_mat["weight"].reshape(B, 1, N)

            output_shape = (B, C, self.H*self.W)
            output_tensor = torch.zeros(output_shape, dtype=residual.dtype, device=residual.device)
            
            # Create the indices tensor for scattering
            indices = sp_mat["cell_ind"].unsqueeze(1).expand(-1, C, -1)
            
            # Scatter the points
            residual = torch.scatter(output_tensor, 2, indices, residual.float(), reduce='add')


        residual = residual.reshape(B, C, self.H, self.W)
        # FFN
        residual = self.ffn(residual)
        # Inflate
        residual = residual.reshape(B, C, self.H * self.W)
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)

    def forward(self, tokens, sp_mat, skip):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        if self.compressed:
            return self.forward_compressed(tokens, sp_mat)
        

        B, C, N = tokens.shape
        residual = self.norm(tokens)
        #feats = self.conv(residual) # parallel 3d branch on point cloud

        # Flatten
        with autocast("cuda", enabled=False):

            residual = residual * sp_mat["weight"].reshape(B, 1, N)

            output_shape = (B, C, self.H*self.W)
            output_tensor = torch.zeros(output_shape, dtype=residual.dtype, device=residual.device)
            
            # Create the indices tensor for scattering
            indices = sp_mat["cell_ind"].unsqueeze(1).expand(-1, C, -1)
            
            # Scatter the points
            residual = torch.scatter_add(output_tensor, 2, indices, residual.float())

        residual = residual.reshape(B, C, self.H, self.W)
        # skip connection with previous layer
        residual = residual + skip
        # FFN
        residual_plane = self.ffn(residual)
        # LayerScale
        residual = residual_plane.reshape(B, C, self.H * self.W)
        residual = self.att(residual) * residual
        # Inflate
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual), residual_plane

class Backbone(nn.Module):
    def __init__(self, channels, depth, grids_shape, drop_path_prob, layer_norm=False):
        super().__init__()
        self.depth = depth
        self.grids_shape = grids_shape
        print("WaffleAndRange grids shape", self.grids_shape)
        self.channel_mix = nn.ModuleList(
            [ChannelMix(channels, drop_path_prob, layer_norm) for _ in range(depth)]
        )
        self.spatial_mix = nn.ModuleList(
            [
                SpatialMix(channels, grids_shape[d % len(grids_shape)], drop_path_prob, layer_norm)
                for d in range(depth)
            ]
        )

    def compress(self):
        for d in range(self.depth):
            self.channel_mix[d].compress()
            self.spatial_mix[d].compress()

    def forward(self, tokens, cell_ind, occupied_cell):
        # Build projection matrices
        batch_size, num_points = tokens.shape[0], tokens.shape[-1]
        point_ind = (
            torch.arange(num_points, device=tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(1, -1)
        )
        batch_ind = (
            torch.arange(batch_size, device=tokens.device)
            .unsqueeze(1)
            .expand(-1, num_points)
            .reshape(1, -1)
        )
        non_zeros_ind = []
        for i in range(cell_ind.shape[1]):
            non_zeros_ind.append(
                torch.cat((batch_ind, cell_ind[:, i].reshape(1, -1), point_ind), axis=0)
            )
        sp_mat = [
            build_proj_matrix(
                id,
                occupied_cell,
                batch_size,
                np.prod(sh),
                cell_ind[:, i],
                tokens.shape[1],
            )
            for i, (id, sh) in enumerate(zip(non_zeros_ind, self.grids_shape))
        ]

        skip = [torch.zeros(size).cuda() for size in self.grids_shape]

        # Actual backbone
        for d, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            tokens, res = smix(tokens, sp_mat[d % len(sp_mat)], skip[d % len(sp_mat)])
            tokens = cmix(tokens)
            # update skip connection
            skip[d % len(sp_mat)] = res

        return tokens
