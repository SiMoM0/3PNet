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


from .pc_dataset import Collate
from .semantic_kitti import SemanticKITTI
from .semantic_kitti_sdata import SemanticKITTIsd # small data setup
from .pandaset import PandaSet

__all__ = [SemanticKITTI, Collate]
LIST_DATASETS = {"semantic_kitti": SemanticKITTI, "semantic_kitti_sdata": SemanticKITTIsd, "pandaset": PandaSet}
