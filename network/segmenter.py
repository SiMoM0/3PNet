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


import torch.nn as nn
from .backbone import Backbone
from .embedding import Embedding

class Segmenter(nn.Module):
    def __init__(
        self,
        input_channels,
        feat_channels,
        nb_class,
        depth,
        grid_shape,
        drop_path_prob=0,
        layer_norm=False,
    ):
        super().__init__()
        # Embedding layer
        self.embed = Embedding(input_channels, feat_channels)
        # 3PNet backbone
        self.waffleiron = Backbone(feat_channels, depth, grid_shape, drop_path_prob, layer_norm)
        # Classification layer
        self.classif = nn.Conv1d(feat_channels, nb_class, 1)

    def compress(self):
        self.embed.compress()
        self.waffleiron.compress()

    def forward(self, feats, cell_ind, occupied_cell, neighbors):
        tokens, local_features = self.embed(feats, neighbors)
        tokens = self.waffleiron(tokens, cell_ind, occupied_cell)

        return self.classif(tokens + local_features) # skip connection with neighbors
        #return self.classif(tokens) # only for semantickitti small data and pandaset