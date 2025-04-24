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


import os
import yaml
import torch
import warnings
import numpy as np
from glob import glob
from tqdm import tqdm
import utils.transforms as tr
from .pc_dataset import PCDataset


class InstanceCutMix:
    def __init__(self, phase="train", temp_dir="/tmp/semantic_kitti_instances/"):
        # Train or Trainval
        self.phase = phase
        assert self.phase in ["train", "trainval"]

        # List of files containing instances for bicycle, motorcycle, person, bicyclist
        self.bank = {1: [], 2: [], 4: [], 5: [], 6: []}

        # Directory where to store instances
        self.rootdir = os.path.join(temp_dir, self.phase)
        for id_class in self.bank.keys():
            os.makedirs(os.path.join(self.rootdir, f"{id_class}"), exist_ok=True)

        # Load instances
        for key in self.bank.keys():
            self.bank[key] = glob(os.path.join(self.rootdir, f"{key}", "*.bin"))
        self.__loaded__ = self.test_loaded()
        if not self.__loaded__:
            warnings.warn(
                "Instances must be extracted and saved on disk before training"
            )

        # Augmentations applied on Instances
        self.rot = tr.Compose(
            (
                tr.FlipXY(inplace=True),
                tr.Rotation(inplace=True),
                tr.Scale(dims=(0, 1, 2), range=0.1, inplace=True),
            )
        )

        # For each class, maximum number of instance to add
        self.num_to_add = 40 # default 40

        # Voxelization of 1m to downsample point cloud to ensure that
        # center of the instances are at least 1m away
        self.vox = tr.Voxelize(dims=(0, 1, 2), voxel_size=1.0, random=True)

    def test_loaded(self):
        self.__loaded__ = False
        if self.phase == "train":
            if len(self.bank[1]) != 5083:
                return False
            if len(self.bank[2]) != 3092:
                return False
            if len(self.bank[4]) != 7419:
                return False
            if len(self.bank[5]) != 8084:
                return False
            if len(self.bank[6]) != 1551:
                return False
        elif self.phase == "trainval":
            if len(self.bank[1]) != 8213:
                return False
            if len(self.bank[2]) != 4169:
                return False
            if len(self.bank[4]) != 10516:
                return False
            if len(self.bank[5]) != 12190:
                return False
            if len(self.bank[6]) != 2943:
                return False
        self.__loaded__ = True
        return True

    def beam_upsample(self, pc, labels, rate=0.1, voxel_size=0.05):
        '''
        Beam upsampling adding lines of points
        '''
        unique_z = np.unique((pc[:, 2] / voxel_size).astype('int'))
        # randomly select a few beams to add
        beam_to_add = np.random.choice(unique_z, int(len(unique_z) * rate), replace=False)

        new_pc = pc
        for beam in beam_to_add:
            beam_points = pc[(pc[:, 2] / voxel_size).astype('int') == beam]
            new_points = np.zeros_like(beam_points)
            new_points[:, 2] = beam_points[:, 2] * np.random.uniform(0.9, 1.1)
            new_points[:, :2] = beam_points[:, :2]
            new_pc = np.concatenate([new_pc, new_points], axis=0)
        new_labels = np.ones((new_pc.shape[0], ), dtype=np.int64) * labels[0]
        return new_pc, new_labels
    
    def beam_downsample(self, pc, labels, rate=0.2, voxel_size=0.05):
        '''
        Beam downsampling removing lines of points
        '''
        unique_z = np.unique((pc[:, 2] / voxel_size).astype('int'))
        # randomly select a few beams to remove
        beam_to_remove = np.random.choice(unique_z, int(len(unique_z) * rate), replace=False)
        new_pc = pc[~np.isin(pc[:, 2] // voxel_size, beam_to_remove)]
        new_labels = np.ones((new_pc.shape[0], ), dtype=np.int64) * labels[0]
        return new_pc, new_labels

    def cut(self, pc, class_label, instance_label):
        for id_class in self.bank.keys():
            where_class = class_label == id_class
            all_instances = np.unique(instance_label[where_class])
            for id_instance in all_instances:
                # Segment instance
                where_ins = instance_label == id_instance
                if where_ins.sum() <= 5:
                    continue
                instance = pc[where_ins, :]
                # Center instance
                instance[:, :2] -= instance[:, :2].mean(0, keepdims=True)
                instance[:, 2] -= instance[:, 2].min(0, keepdims=True)
                # Save instance
                pathfile = os.path.join(
                    self.rootdir, f"{id_class}", f"{len(self.bank[id_class]):07d}.bin"
                )
                instance.tofile(pathfile) # TODO: save also info about laser:id for each point of the instance
                self.bank[id_class].append(pathfile)

    def mix(self, pc, class_label):
        # Find potential location where to add new object (on a surface)
        pc_vox, class_label_vox = self.vox(pc, class_label)
        where_surface = np.where((class_label_vox >= 8) & (class_label_vox <= 10))[0]
        where_surface = where_surface[torch.randperm(len(where_surface))]

        distance = np.linalg.norm(pc[:, :2], axis=1)

        # Add instances of each class in bank
        id_tot = 0
        new_pc, new_label = [pc], [class_label]
        for id_class in self.bank.keys():
            nb_to_add = torch.randint(self.num_to_add, (1,))[0]
            nb_to_add = np.min((nb_to_add, len(where_surface) - id_tot))
            which_one = torch.randint(len(self.bank[id_class]), (nb_to_add,))
            for ii in range(nb_to_add):
                # Point p where to add the instance
                p = pc_vox[where_surface[id_tot]]
                p_range = np.linalg.norm(p[:2])
                # Extract instance
                object = self.bank[id_class][which_one[ii]]
                object = np.fromfile(object, dtype=np.float32).reshape((-1, 4))
                # Augment instance
                label = np.ones((object.shape[0],), dtype=np.int64) * id_class
                object, label = self.rot(object, label)
                # INSTANCE UPSAMPLING/DOWNSAMPLING BASED ON RANGE
                ref_distance = (distance.max() - distance.min()) / 3    # reference to switch between upsample and downsample
                if p_range < ref_distance:
                    rate = (1 - (p_range - distance.min()) / (ref_distance - distance.min())) / 2
                    object, label = self.beam_upsample(object, label, rate)
                elif p_range > ref_distance:
                    rate = ((p_range - ref_distance) / (distance.max() - ref_distance)) / 2
                    object, label = self.beam_downsample(object, label, rate)
                # Move instance at point p
                object[:, :3] += p[:3][None]
                # Add instance in the point cloud
                new_pc.append(object)
                # Add corresponding label
                new_label.append(label)
                id_tot += 1

        return np.concatenate(new_pc, 0), np.concatenate(new_label, 0)

    def __call__(self, pc, class_label, instance_label):
        if not self.__loaded__:
            self.cut(pc, class_label, instance_label)
            return None, None
        return self.mix(pc, class_label)


class PolarMix:
    def __init__(self, classes=None, inst=True):
        self.classes = classes
        self.inst = inst
        self.rot = tr.Rotation(inplace=False)

    def __call__(self, pc1, label1, pc2, label2):
        # --- Scene-level swapping
        if torch.rand(1)[0] < 0.5:
            sector = (2 * float(torch.rand(1)[0]) - 1) * np.pi
            # --- Remove a 180 deg sector in 1
            theta1 = (np.arctan2(pc1[:, 1], pc1[:, 0]) - sector) % (2 * np.pi)
            where1 = (theta1 > 0) & (theta1 < np.pi)
            where1 = ~where1
            pc1, label1 = pc1[where1], label1[where1]
            # --- Replace by corresponding 180 deg sector in 2
            theta2 = (np.arctan2(pc2[:, 1], pc2[:, 0]) - sector) % (2 * np.pi)
            where2 = (theta2 > 0) & (theta2 < np.pi)
            #
            pc = np.concatenate((pc1, pc2[where2]), axis=0)
            label = np.concatenate((label1, label2[where2]), axis=0)
        else:
            pc, label = pc1, label1

        if self.inst:
            # --- Instance level augmentation
            where_class = label2 == self.classes[0]
            for id_class in self.classes[1:]:
                where_class |= label2 == id_class
            if where_class.sum() > 0:
                pc2, label2 = pc2[where_class], label2[where_class]
                pc22 = self.rot(pc2, label2)[0]
                pc23 = self.rot(pc2, label2)[0]
                pc = np.concatenate((pc, pc2, pc22, pc23), axis=0)
                label = np.concatenate((label, label2, label2, label2), axis=0)

        return pc, label


class PandaSet(PCDataset):
    CLASS_NAME = [
        "car",  # 0
        "bicycle",  # 1
        "motorcycle",  # 2
        "truck",  # 3
        "other-vehicle",  # 4
        "person",  # 5
        "road",  # 6
        "road barriers",  # 7
        "sidewalk",  # 8
        "building",  # 9
        "vegetation",  # 10
        "terrain",  # 11
        "background",  # 12
        "traffic-sign",  # 13
        "ignore",  # 14
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Config file and class mapping
        current_folder = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_folder, "pandaset.yaml")) as stream: # use pandaset.yaml or pandaset-sdata.yaml
            pandasetyaml = yaml.safe_load(stream)
        self.learning_map = pandasetyaml["learning_map"]

        # Split
        if self.phase == "train":
            split = pandasetyaml["split"]["train"]
        elif self.phase == "val":
            split = pandasetyaml["split"]["valid"]
        elif self.phase == "test":
            split = pandasetyaml["split"]["test"]
        elif self.phase == "trainval":
            split = pandasetyaml["split"]["train"] + pandasetyaml["split"]["valid"]
        else:
            raise Exception(f"Unknown split {self.phase}")

        # Find all files
        self.im_idx = []
        for i_folder in np.sort(split):
            self.im_idx.extend(
                glob(
                    os.path.join(
                        self.rootdir,   # based on PandaSet converted to KITTI format
                        #"dataset",
                        #"sequences",
                        str(i_folder).zfill(3),
                        "velodyne",
                        "*.bin",
                    )
                )
            )
        
        self.im_idx = np.sort(self.im_idx)

        # Training with instance cutmix
        if self.instance_cutmix:
            # PolarMix
            self.polarmix = PolarMix(classes=[1, 2, 4, 5, 6])
            # CutMix
            assert (
                self.phase != "test" and self.phase != "val"
            ), "Instance cutmix should not be applied at test or val time"
            self.cutmix = InstanceCutMix(phase=self.phase)
            if not self.cutmix.test_loaded():
                print("Extracting instances before training...")
                for index in tqdm(range(len(self))):
                    self.load_pc(index)
                print("Done.")
            assert self.cutmix.test_loaded(), "Instances not extracted correctly"

        # Training with only polarmix
        if self.polarmix_only:
            self.polarmix = PolarMix(classes=None, inst=False)

    def __len__(self):
        return len(self.im_idx)

    def __load_pc_internal__(self, index):
        # Load point cloud
        pc = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 5))

        # Extract Label
        labels_inst = np.fromfile(
            self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
            dtype=np.uint32,
        ).reshape((-1, 1))
        #labels = labels_inst & 0xFFFF  # delete high 16 digits binary
        labels = np.vectorize(self.learning_map.__getitem__)(labels_inst).astype(
            np.int32
        )

        # Map ignore index 14 (unlabelled) to 255
        labels = labels[:, 0]
        labels[labels == 14] = 255

        return pc, labels, labels_inst[:, 0]

    def load_pc(self, index):
        pc, labels, labels_inst = self.__load_pc_internal__(index)

        # Instance CutMix and Polarmix
        if self.instance_cutmix:
            # Polarmix
            if self.cutmix.test_loaded():
                new_index = torch.randint(len(self), (1,))[0]
                new_pc, new_label, _ = self.__load_pc_internal__(new_index)
                pc, labels = self.polarmix(pc, labels, new_pc, new_label)
            # Cutmix
            pc, labels = self.cutmix(pc, labels, labels_inst)

        # PolarMix only
        if self.polarmix_only:
            new_index = torch.randint(len(self), (1,))[0]
            new_pc, new_label, _ = self.__load_pc_internal__(new_index)
            pc, labels = self.polarmix(pc, labels, new_pc, new_label)

        return pc, labels, self.im_idx[index]
