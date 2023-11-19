import os
import torch

import torch.distributed as dist
from monai.data import (CacheDataset, PersistentDataset, DataLoader, load_decathlon_datalist,
                        load_decathlon_properties, partition_dataset)

from transforms import get_task_transforms


def get_data(param, mode="train"):
    # get necessary parameters:

    transform = get_task_transforms(mode, param)

    list_key = mode
    if mode == "test" and param.save_prob_maps:
        list_key = "validation"

    property_keys = [
        "name",
        "description",
        "reference",
        "licence",
        "tensorImageSize",
        "modality",
        "labels",
        "numTraining",
        "numTest",
    ]

    datalist = load_decathlon_datalist(param.dataset_desc, True, list_key, param.data_root)

    properties = load_decathlon_properties(param.dataset_desc, property_keys)
    if mode == "test":
        if param.multi_gpu:
            datalist = partition_dataset(
                data=datalist,
                shuffle=False,
                num_partitions=dist.get_world_size(),
                even_divisible=False,
            )[dist.get_rank()]

        if param.ds_cache_dir is not None:
            val_ds = PersistentDataset(
                data=datalist,
                transform=transform,
                cache_dir=param.ds_cache_dir
            )
        else:
            val_ds = CacheDataset(
                data=datalist,
                transform=transform,
                num_workers=4,
            )

        data_loader = DataLoader(
            val_ds,
            # slw_batchsize is set outside
            batch_size=1,
            shuffle=False,
            num_workers=param.val_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    elif mode in ["train", "validation"]:
        if param.multi_gpu:
            datalist = partition_dataset(
                data=datalist,
                shuffle=True,
                num_partitions=dist.get_world_size(),
                even_divisible=True,
            )[dist.get_rank()]

        if param.ds_cache_dir is not None:
            train_ds = PersistentDataset(
                data=datalist,
                transform=transform,
                cache_dir=param.ds_cache_dir
            )
        else:        
            train_ds = CacheDataset(
                data=datalist,
                transform=transform,
                num_workers=8,
                cache_rate=param.ds_cache_rate,
            )

        data_loader = DataLoader(
            train_ds,
            batch_size=param.batch_size,
            shuffle=True,
            num_workers=param.num_workers if mode == "train" else param.val_num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        raise ValueError(f"mode should be train, validation or test.")

    return properties, data_loader
