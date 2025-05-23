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


import os
import yaml
import torch
import argparse
import network
import numpy as np
from tqdm import tqdm
from network.segmenter import Segmenter
from datasets import SemanticKITTI, Collate

import cupy as cp
import time

import ctypes
from ctypes import cdll
from numpy.ctypeslib import ndpointer

lib = cdll.LoadLibrary('cudastuff/build/libmylib_user.so')
obj = lib.MyFastKDTree_new()

runTree = lib.MyFastKDTree_run

runTree.argtypes = [ctypes.c_void_p,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  ## indatav
        ctypes.c_size_t,                                  ## tree_size  
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  ## queries_k
        ctypes.c_size_t,                                  ## numQueries_k
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  ## queries_1
        ctypes.c_size_t,                                  ## numQueries_1
        ctypes.POINTER(ctypes.c_int),                     ## d_results_k
        ctypes.POINTER(ctypes.c_int),]                    ## d_results_1

runTree.restype   = ctypes.c_void_p


def distributed_training(gpu, ngpus_per_node, args, config, remap_lut):
    # --- Init. distributing training
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    args.rank = args.rank * ngpus_per_node + gpu
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # --- Build network
    net = Segmenter(
        input_channels=config["embedding"]["size_input"],
        feat_channels=config["waffleiron"]["nb_channels"],
        depth=config["waffleiron"]["depth"],
        grid_shape=config["waffleiron"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
        drop_path_prob=config["waffleiron"]["drop"],
    )
    net = net.cuda()

    # --- Load weights
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    try:
        net.load_state_dict(ckpt["net"])
    except:
        # If net was trained using DataParallel or DistributedDataParallel
        state_dict = {}
        for key in ckpt["net"].keys():
            state_dict[key[len("module."):]] = ckpt["net"][key]
        net.load_state_dict(state_dict, strict=False)
    #net.compress()
    net.eval()

    # ---
    #args.batch_size = config["dataloader"]["batch_size"]
    #args.workers = config["dataloader"]["num_workers"]

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(args.gpu)
    net.cuda(args.gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs of the current node.
    #args.batch_size = int(config["dataloader"]["batch_size"] / ngpus_per_node)
    args.workers = int(
        (config["dataloader"]["num_workers"] + ngpus_per_node - 1) / ngpus_per_node
    )
    #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    #net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    
    if args.gpu == 0 or args.gpu is None:
        #print(f"net:\n{net}")
        nb_param = sum([p.numel() for p in net.parameters()]) / 1e6
        print(f"{nb_param} x 10^6 trainable parameters ")

    # --- Dataloader
    tta = args.num_votes > 1
    dataset = SemanticKITTI(
        rootdir=args.path_dataset,
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors=config["embedding"]["neighbors"],
        dim_proj=config["waffleiron"]["dim_proj"],
        grids_shape=config["waffleiron"]["grids_size"],
        fov_xyz=config["waffleiron"]["fov_xyz"],
        phase=args.phase,
        tta=tta,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) ## NEW
    sampler.set_epoch(0)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler, ## NEW
        drop_last=False,
        collate_fn=Collate(),
    )
    args.num_votes = args.num_votes // args.batch_size

    soft = torch.nn.Softmax(dim=1).cuda()
    
    # --- Re-activate droppath if voting
    if tta:
        for m in net.modules():
            if isinstance(m, network.backbone.DropPath):
                m.train()

    # --- Evaluation
    id_vote = 0
    for it, batch in enumerate(
        tqdm(loader, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
    ):
        # Reset vote
        if id_vote == 0:
            vote = None


        neigs = []
        upsample = []
        for pc, pc_orig in zip(batch["pc"], batch["pc_orig"]):
            N = pc.shape[0]

            data = pc[:, :3].reshape(-1).astype(np.float32)
            results_k = cp.zeros((N, 17), dtype=cp.int32)
            results_1 = cp.zeros(pc_orig.shape[0], dtype=cp.int32)

            results_k_ctypes = ctypes.cast(results_k.data.ptr, ctypes.POINTER(ctypes.c_int32))
            results_1_ctypes = ctypes.cast(results_1.data.ptr, ctypes.POINTER(ctypes.c_int32))

            full = pc_orig[:, :3].reshape(-1).astype(np.float32)

            runTree(obj, data, data.size, data, data.size, full, full.shape[0], results_k_ctypes, results_1_ctypes)
            results_k = torch.from_dlpack(results_k.toDlpack()).long()
            results_1 = torch.from_dlpack(results_1.toDlpack()).long()
            
            new_arr = torch.argsort(results_k[:, 0])

            neigs.append(new_arr[results_k].T[None])
            upsample.append(new_arr[results_1.T])

        neighbors_emb = torch.vstack(neigs).long()

        # Network inputs
        feat = batch["feat"].cuda(non_blocking=True)
        # labels = batch["labels_orig"].cuda(non_blocking=True)
        cell_ind = batch["cell_ind"].cuda(non_blocking=True)
        occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # Get prediction
        with torch.autocast("cuda", enabled=True):
            with torch.inference_mode():
                # Get prediction
                # torch.cuda.synchronize()
                # start = time.time()
                out = net(*net_inputs)
                # torch.cuda.synchronize()
                # end = time.time()
                # print(f"Model Time: {end - start:.3f}")
                for b in range(out.shape[0]):
                    temp = out[b, :, upsample[b]].T
                    if vote is None:
                        vote = soft(temp)
                    else:
                        vote += soft(temp)
        id_vote += 1

        # Save prediction
        if id_vote == args.num_votes:
            # Convert label
            pred_label = (
                vote.max(1)[1] + 1
            )  # Shift by 1 because of ignore_label at index 0
            label = pred_label.cpu().numpy().reshape(-1).astype(np.uint32)
            upper_half = label >> 16  # get upper half for instances
            lower_half = label & 0xFFFF  # get lower half for semantics
            lower_half = remap_lut[lower_half]  # do the remapping of semantics
            label = (upper_half << 16) + lower_half  # reconstruct full label
            label = label.astype(np.uint32)
            # Save result
            assert batch["filename"][0] == batch["filename"][-1]
            label_file = batch["filename"][0][
                len(os.path.join(dataset.rootdir, "dataset/")):
            ]
            label_file = label_file.replace("velodyne", "predictions")[:-3] + "label"
            label_file = os.path.join(args.result_folder, label_file)

            os.makedirs(os.path.split(label_file)[0], exist_ok=True)
            label.tofile(label_file)
            # Reset count of votes
            id_vote = 0




if __name__ == "__main__":
    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint")
    parser.add_argument(
        "--path_dataset", type=str, help="Path to SemanticKITTI dataset"
    )
    parser.add_argument("--result_folder", type=str, help="Path to where result folder")
    parser.add_argument(
        "--num_votes", type=int, default=1, help="Number of test time augmentations"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--phase", required=True, help="val or test")
    args = parser.parse_args()
    #assert args.num_votes % args.batch_size == 0
    os.makedirs(args.result_folder, exist_ok=True)

    # --- Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- SemanticKITTI (from https://github.com/PRBonn/semantic-kitti-api/blob/master/remap_semantic_labels.py)
    with open("./datasets/semantic-kitti.yaml") as stream:
        semkittiyaml = yaml.safe_load(stream)
    remapdict = semkittiyaml["learning_map_inv"]
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    """ -------------------NEW------------------- """
    if args.num_workers>=0:
        args.rank = 0
        # Number of nodes for distributed training'
        args.world_size = 1
        # URL used to set up distributed training
        args.dist_url = "tcp://127.0.0.1:4445"
        # Distributed backend'
        args.dist_backend = "nccl"
        ngpus_per_node = torch.cuda.device_count()

        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(
                distributed_training,
                nprocs=ngpus_per_node,
                args=(ngpus_per_node, args, config, remap_lut),
            )
        """ ----------------------------------------- """

    else:


        # --- Dataloader
        tta = args.num_votes > 1
        dataset = SemanticKITTI(
            rootdir=args.path_dataset,
            input_feat=config["embedding"]["input_feat"],
            voxel_size=config["embedding"]["voxel_size"],
            num_neighbors=config["embedding"]["neighbors"],
            dim_proj=config["waffleiron"]["dim_proj"],
            grids_shape=config["waffleiron"]["grids_size"],
            fov_xyz=config["waffleiron"]["fov_xyz"],
            phase=args.phase,
            tta=tta,
        )
        if args.num_votes > 1:
            new_list = []
            for f in dataset.im_idx:
                for v in range(args.num_votes):
                    new_list.append(f)
            dataset.im_idx = new_list

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=Collate(),
        )
        args.num_votes = args.num_votes // args.batch_size
        # --- Build network
        net = Segmenter(
            input_channels=config["embedding"]["size_input"],
            feat_channels=config["waffleiron"]["nb_channels"],
            depth=config["waffleiron"]["depth"],
            grid_shape=config["waffleiron"]["grids_size"],
            nb_class=config["classif"]["nb_class"],
            drop_path_prob=config["waffleiron"]["drop"],
        )
        net = net.cuda()

        # --- Load weights
        ckpt = torch.load(args.ckpt, map_location="cuda:0")
        try:
            net.load_state_dict(ckpt["net"])
        except:
            # If net was trained using DataParallel or DistributedDataParallel
            state_dict = {}
            for key in ckpt["net"].keys():
                state_dict[key[len("module."):]] = ckpt["net"][key]
            net.load_state_dict(state_dict, strict=False)
        #net.compress()
        net.eval()

        

        # --- Re-activate droppath if voting
        if tta:
            for m in net.modules():
                if isinstance(m, network.backbone.DropPath):
                    m.train()
        

        # --- Evaluation
        id_vote = 0
        for it, batch in enumerate(
            tqdm(loader, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
        ):
            # Reset vote
            if id_vote == 0:
                vote = None

            Nmax = np.max([f.shape[0] for f in batch["pc"]])
            neigs = []
            print("Nmax", Nmax)
            for pc in batch["pc"]:
                N = pc.shape[0]
                data =pc[:, :3].reshape(-1).astype(np.float32)

                s = time.time()
                buildTree(obj, data, data.size)
                e = time.time()
                print("first build time", e-s)

                a = np.zeros((N, 17), dtype=np.int32)
                input_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                s = time.time()
                queryTree(obj, data, data.size, input_ptr)

                e = time.time()
                print("query time", e-s)
                new_arr = np.argsort(a[:, 0])
                neighbors_emb = new_arr[a].T[None]
                neighbors_emb = np.concatenate(
                    (
                        neighbors_emb,
                        (Nmax - 1) * np.ones((1, neighbors_emb.shape[1], Nmax - N)),
                    ),
                    axis=2,
                )
                neigs.append(neighbors_emb)
            neighbors_emb = torch.from_numpy(np.vstack(neigs)).long()  # B x Nmax
            neighbors_emb=neighbors_emb.cuda(non_blocking=True)

            print("neighbors_emb", neighbors_emb.shape)

            s = time.time()
            upsample = []
            for pc_orig in batch["pc_orig"]:
                full = pc_orig[:, :3].reshape(-1).astype(np.float32)

                a = np.zeros(pc_orig.shape[0], dtype=np.int32)
                input_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                queryTree_1(obj, full, full.shape[0], input_ptr)
                upsample_ = new_arr[a.T]
                upsample.append(torch.from_numpy(upsample_))
            e = time.time()
            print("second query time", e-s)


            # Network inputs
            feat = batch["feat"].cuda(non_blocking=True)
            labels = batch["labels_orig"].cuda(non_blocking=True)
            #batch["upsample"] = [up.cuda(non_blocking=True) for up in batch["upsample"]]
            cell_ind = batch["cell_ind"].cuda(non_blocking=True)
            occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
            #neighbors_emb = batch["neighbors_emb"].cuda(non_blocking=True)
            net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

            ### forse non necessario!!
            #pc_orig = [pc_orig_.cuda(non_blocking=True) for pc_orig_ in batch["pc_orig"]]

            # Get prediction
            with torch.autocast("cuda", enabled=True):
                with torch.inference_mode():
                    # Get prediction
                    s = time.time()
                    out = net(*net_inputs)
                    e = time.time()
                    print("latency", e-s)
                    for b in range(out.shape[0]):
                        temp = out[b, :, upsample[b]].T
                        if vote is None:
                            vote = torch.softmax(temp, dim=1)
                        else:
                            vote += torch.softmax(temp, dim=1)
            id_vote += 1
            
            # Save prediction
            if id_vote == args.num_votes:
                # Convert label
                pred_label = (
                    vote.max(1)[1] + 1
                )  # Shift by 1 because of ignore_label at index 0
                label = pred_label.cpu().numpy().reshape(-1).astype(np.uint32)
                upper_half = label >> 16  # get upper half for instances
                lower_half = label & 0xFFFF  # get lower half for semantics
                lower_half = remap_lut[lower_half]  # do the remapping of semantics
                label = (upper_half << 16) + lower_half  # reconstruct full label
                label = label.astype(np.uint32)
                # Save result
                assert batch["filename"][0] == batch["filename"][-1]
                label_file = batch["filename"][0][
                    len(os.path.join(dataset.rootdir, "dataset/")):
                ]
                label_file = label_file.replace("velodyne", "predictions")[:-3] + "label"
                label_file = os.path.join(args.result_folder, label_file)
                os.makedirs(os.path.split(label_file)[0], exist_ok=True)
                label.tofile(label_file)
                # Reset count of votes