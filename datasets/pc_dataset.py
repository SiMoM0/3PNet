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
import numpy as np
import utils.transforms as tr
from torch.utils.data import Dataset
import time
from scipy.spatial import cKDTree as KDTree

class PCDataset(Dataset):
    def __init__(
        self,
        rootdir=None,
        phase="train",
        input_feat="intensity",
        voxel_size=0.1,
        train_augmentations=None,
        dim_proj=[
            0,
        ],
        grids_shape=[(256, 256)],
        fov_xyz=(
            (
                -1.0,
                -1.0,
                -1.0,
            ),
            (1.0, 1.0, 1.0),
        ),
        num_neighbors=16,
        tta=False,
        instance_cutmix=False,
        polarmix=False,
    ):
        super().__init__()

        # Dataset split
        self.phase = phase
        assert self.phase in ["train", "val", "trainval", "test"]

        # Root directory of dataset
        self.rootdir = rootdir

        # Input features to compute for each point
        self.input_feat = input_feat

        # Downsample input point cloud by small voxelization
        self.downsample = tr.Voxelize(
            dims=(0, 1, 2),
            voxel_size=voxel_size,
            random=(self.phase == "train" or self.phase == "trainval"),
        )

        # Field of view
        assert len(fov_xyz[0]) == len(
            fov_xyz[1]
        ), "Min and Max FOV must have the same length."
        for i, (min, max) in enumerate(zip(*fov_xyz)):
            assert (
                min < max
            ), f"Field of view: min ({min}) < max ({max}) is expected on dimension {i}."
        self.fov_xyz = np.concatenate([np.array(f)[None] for f in fov_xyz], axis=0)
        self.crop_to_fov = tr.Crop(dims=(0, 1, 2), fov=fov_xyz)

        # Grid shape for projection in 2D
        assert len(grids_shape) == len(dim_proj)
        self.dim_proj = dim_proj
        self.grids_shape = [np.array(g) for g in grids_shape]
        self.lut_axis_plane = {1: (1, 2), 2: (0, 2), 3: (0, 1)}

        # Number of neighbors for embedding layer
        assert num_neighbors > 0
        self.num_neighbors = num_neighbors

        # Test time augmentation
        if tta:
            assert self.phase in ["test", "val"]
            self.tta = tr.Compose(
                (
                    tr.Rotation(inplace=True, dim=2),
                    tr.Rotation(inplace=True, dim=6),
                    tr.RandomApply(tr.FlipXY(inplace=True), prob=2.0 / 3.0),
                    tr.Scale(inplace=True, dims=(0, 1, 2), range=0.1),
                )
            )
        else:
            self.tta = None

        # Train time augmentations
        if train_augmentations is not None:
            assert self.phase in ["train", "trainval"]
        self.train_augmentations = train_augmentations

        # Flag for instance cutmix
        self.instance_cutmix = instance_cutmix

        # Flag for only polar mix
        self.polarmix_only = polarmix
        
    def do_range_projection(self, pc, proj_H=64, proj_W=1024, proj_fov_up=15, proj_fov_down=-25, laser_id=None):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad


        # get scan components
        scan_x = pc[:, 0]
        scan_y = pc[:, 1]
        scan_z = pc[:, 2]
        depth = np.linalg.norm(pc[:, :3], 2, axis=1)

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= proj_W                              # in [0.0, W]
        proj_y *= proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

        # use laser id for PandaSet
        if laser_id is not None:
            proj_y = laser_id.astype(np.int32)

        return proj_y * proj_W + proj_x
    
    def do_polar_projection(self, pc, num_rings=64, num_sectors=1024):
        """ Project a pointcloud into a polar grid and unroll it.
        """
        # calculate polar coordinates
        rho = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        phi = np.arctan2(pc[:, 1], pc[:, 0])

        # get min and max rho
        rho_min = np.min(rho)
        rho_max = np.max(rho)

        # straightforward way to sicretize the grid
        #r = (rho / np.max(rho) * num_rings - 1).astype(int) # linear scale
        #r = ((rho - rho_min) / (rho.max() - rho_min) * (num_rings - 1)).astype(int) # customize linear scale
        #r = (np.log(rho + 1e-6) / np.log(rho.max() + 1e-6) * (num_rings - 1)).astype(int) # log scale
        r = ((np.log(rho + 1e-6) - np.log(rho_min + 1e-6)) / (np.log(rho_max + 1e-6) - np.log(rho_min + 1e-6)) * (num_rings - 1)).astype(int) # customize log scale
        s = ((phi + np.pi) / (2 * np.pi) * num_sectors - 1).astype(int)

        return r * num_sectors + s
        

    def get_occupied_2d_cells(self, pc, laser_id=None):
        """Return mapping between 3D point and corresponding 2D cell"""

        cell_ind = []
        for dim, grid in zip(self.dim_proj[1:-1], self.grids_shape[1:-1]):  # Skip polar grid and range image
            # Get plane of which to project
            dims = self.lut_axis_plane[dim]
            # Compute grid resolution
            res = (self.fov_xyz[1, dims] - self.fov_xyz[0, dims]) / grid[None]
            # Shift and quantize point cloud
            pc_quant = ((pc[:, dims] - self.fov_xyz[0, dims]) / res).astype("int")
            # Check that the point cloud fits on the grid
            min, max = pc_quant.min(0), pc_quant.max(0)
            assert min[0] >= 0 and min[1] >= 0, print(
                "Some points are outside the FOV:", pc[:, :3].min(0), self.fov_xyz
            )
            assert max[0] < grid[0] and max[1] < grid[1], print(
                "Some points are outside the FOV:", pc[:, :3].min(0), self.fov_xyz
            )
            # Transform quantized coordinates to cell indices for projection on 2D plane
            temp = pc_quant[:, 0] * grid[1] + pc_quant[:, 1]
            cell_ind.append(temp[None])

        range_image = self.do_range_projection(pc, proj_H=self.grids_shape[-1][0], proj_W=self.grids_shape[-1][1], proj_fov_up=3.0, proj_fov_down=-25.0, laser_id=laser_id) # semantickitti and pandaset
        # range image as last projection
        cell_ind.append(range_image[None])

        # polar grid as first projection
        polar_grid = self.do_polar_projection(pc, num_rings=self.grids_shape[0][0], num_sectors=self.grids_shape[0][1])
        cell_ind.insert(0, polar_grid[None])

        return np.vstack(cell_ind)

    def prepare_input_features(self, pc_orig):
        # Concatenate desired input features to coordinates
        pc = [pc_orig[:, :3]]  # Initialize with coordinates
        for type in self.input_feat:
            if type == "intensity":
                pc.append(pc_orig[:, 3:4])
            elif type == "height":
                pc.append(pc_orig[:, 2:3])
            elif type == "radius":
                r_xyz = np.linalg.norm(pc_orig[:, :3], axis=1, keepdims=True)
                pc.append(r_xyz)
            elif type == "xyz":
                pc.append(pc_orig[:, :3])
            else:
                raise ValueError(f"Unknown feature: {type}")
        if pc_orig.shape[1] == 5:
            pc.append(pc_orig[:, 4:]) # laser id for pandaset
        return np.concatenate(pc, 1)

    def load_pc(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        # Load original point cloud
        pc_orig, labels_orig, filename = self.load_pc(index)

        pandaset = True if pc_orig.shape[1] == 5 else False

        # Prepare input feature
        pc_orig = self.prepare_input_features(pc_orig)

        # Test time augmentation
        if self.tta is not None:
            pc_orig, labels_orig = self.tta(pc_orig, labels_orig)

        # Voxelization
        pc, labels = self.downsample(pc_orig, labels_orig)

        # Augment data
        if self.train_augmentations is not None:
            pc, labels = self.train_augmentations(pc, labels)

        # Crop to fov
        pc, labels = self.crop_to_fov(pc, labels)

        if pandaset:
            laser_id = pc[:, -1] # Get laser id for PandaSet
            pc = pc[:, :-1]

        # For each point, get index of corresponding 2D cells on projected grid
        cell_ind = self.get_occupied_2d_cells(pc, laser_id if pandaset else None)

        # Get neighbors for point embedding layer providing tokens to waffleiron backbone
        kdtree = KDTree(pc[:, :3])
        assert pc.shape[0] > self.num_neighbors
        _, neighbors_emb = kdtree.query(pc[:, :3], k=self.num_neighbors + 1)

        # Nearest neighbor interpolation to undo cropping & voxelisation at validation time
        if self.phase in ["train", "trainval"]:
            upsample = np.arange(pc.shape[0])
        else:
            _, upsample = kdtree.query(pc_orig[:, :3], k=1)

        # Output to return
        out = (
            # Point features
            pc[:, 3:].T[None],
            # Point labels of original entire point cloud
            labels if self.phase in ["train", "trainval"] else labels_orig,
            # Projection 2D -> 3D: index of 2D cells for each point
            cell_ind[None],
            # Neighborhood for point embedding layer, which provides tokens to waffleiron backbone
            neighbors_emb.T[None],
            # For interpolation from voxelized & cropped point cloud to original point cloud
            upsample,
            # Filename of original point cloud
            filename,
        )

        return out


def zero_pad(feat, neighbors_emb, cell_ind, Nmax):
    N = feat.shape[-1]
    assert N <= Nmax
    occupied_cells = np.ones((1, Nmax))
    if N < Nmax:
        # Zero-pad with null features
        feat = np.concatenate((feat, np.zeros((1, feat.shape[1], Nmax - N))), axis=2)
                # For zero-padded points, associate last zero-padded points as neighbor

        neighbors_emb = np.concatenate(
            (
                neighbors_emb,
                (Nmax - 1) * np.ones((1, neighbors_emb.shape[1], Nmax - N)),
            ),
            axis=2,
        )

        # Associate zero-padded points to first 2D cell...
        cell_ind = np.concatenate(
            (cell_ind, np.zeros((1, cell_ind.shape[1], Nmax - N))), axis=2
        )
        # ... and at the same time mark zero-padded points as unoccupied
        occupied_cells[:, N:] = 0
    return feat, neighbors_emb, cell_ind, occupied_cells


class Collate:
    def __init__(self, num_points=None):
        self.num_points = num_points
        assert num_points is None or num_points > 0

    def __call__(self, list_data):
        # Extract all data
        list_of_data = (list(data) for data in zip(*list_data))
        feat, label_orig, cell_ind, neighbors_emb, upsample, filename = list_of_data

        # Zero-pad point clouds
        Nmax = np.max([f.shape[-1] for f in feat])
        if self.num_points is not None:
            assert Nmax <= self.num_points
        occupied_cells = []
        for i in range(len(feat)):
            feat[i], neighbors_emb[i], cell_ind[i], temp = zero_pad(
                feat[i],
                neighbors_emb[i],
                cell_ind[i],
                Nmax if self.num_points is None else self.num_points,
            )
            occupied_cells.append(temp)

        # Concatenate along batch dimension
        feat = torch.from_numpy(np.vstack(feat)).float()  # B x C x Nmax
        neighbors_emb = torch.from_numpy(np.vstack(neighbors_emb)).long()  # B x Nmax
        cell_ind = torch.from_numpy(
            np.vstack(cell_ind)
        ).long()  # B x nb_2d_cells x Nmax
        occupied_cells = torch.from_numpy(np.vstack(occupied_cells)).float()  # B x Nmax
        labels_orig = torch.from_numpy(np.hstack(label_orig)).long()
        upsample = [torch.from_numpy(u) for u in upsample]

        # Prepare output variables
        out = {
            "feat": feat,
            "neighbors_emb": neighbors_emb,
            "upsample": upsample,
            "labels_orig": labels_orig,
            "cell_ind": cell_ind,
            "occupied_cells": occupied_cells,
            "filename": filename,
        }

        return out
