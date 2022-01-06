import os
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib import rc

import monai
import torch
import numpy as np

from glob import glob
from monai.data import Dataset, DataLoader
from monai import metrics
from contrib.loss import DiceLoss

from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    BorderPadd,
    RandSpatialCropd,
    AsChannelFirstd,
    RandAffined,
    Spacingd,
    ToTensord,
    ToNumpyd,
    CropForegroundd,
    SpatialPadd,
)

from monai.data import (
    decollate_batch,
)

DATA_FULL = "./data/miccai/full_dataset_nifti/train"
DATA_CUT = "./data/miccai/cut_dataset_nifti/train"

DEFAULT_KEYS = {'img': "volume", 'seg': "segmentation"}


SPACING = [0.98000002, 0.98000002, 2.49962478]

LABEL_IDS = {
    0: "BG",
    1: "BS",
    2: "OC",
    3: "ON_L",
    4: "ON_R",
    5: "PG_L",
    6: "PG_R",
    7: "MA"
}

def create_datalaoder(data: str,
                      patch: list,
                      batch: int = 2,
                      ) -> DataLoader:
    data = os.path.abspath(data)
    volume = sorted(glob(os.path.join(data, '*' + DEFAULT_KEYS["img"] + '.nii.gz')))
    label = sorted(glob(os.path.join(data, '*' + DEFAULT_KEYS["seg"] + '.nii.gz')))

    files = [{"img": v, "seg": l} for v, l in zip(volume, label)]

    transform = [
        LoadImaged(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        Spacingd(keys=['img', 'seg'], pixdim=SPACING,
                 mode=('bilinear', 'nearest')),
    ]


    transform_1 = [
        CropForegroundd(keys=["img", "seg"], source_key="img"),
    ]
    if patch:
        transform_1 += [
            SpatialPadd(keys=["img", "seg"], spatial_size=patch),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg",
                spatial_size=patch, pos=1, neg=2,
                num_samples=batch
            )
        ]


    transform += transform_1
    transforms = Compose(transform)

    ds = Dataset(data=files, transform=transforms)

    # sample repeat * batch patches from an image. (Note samples are different
    # for the each of the nth repetition even though randomness is turned off)
    data_loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        pin_memory=False,
    )# do not use cuda for sampling test

    return data_loader


def eval_loss_functions(dataset, patch, batch=2, pred_class=0):
    loader = create_datalaoder(dataset, patch, batch)

    num_classes = len(LABEL_IDS)

    mydice_scores : list = []
    batchdice_scores : list = []
    dice_scores : list = []
    avg_missing_classes_of_samples : list = []
    missing_classes_of_samples : list = []

    for iter, data in enumerate(loader):

        gt = data['seg']
        gt_one_hot = monai.networks.utils.one_hot(labels=gt,
                                                  num_classes=num_classes)
        pred = torch.zeros_like(gt_one_hot)

        # we are assuming that the network learned a
        pred[:, pred_class, :] = 1

        misses : list = []
        gt_list = decollate_batch(gt)
        for gt_patch in gt_list:
            misses.append(num_classes - len(np.unique(gt_patch.cpu().detach().numpy())))
        avg_batch_miss = np.array(misses).mean()
        sample_miss = num_classes - len(np.unique(gt.cpu().detach().numpy()))
        avg_missing_classes_of_samples.append(avg_batch_miss)
        missing_classes_of_samples.append(sample_miss)

        dice = DiceLoss(to_onehot_y=False, softmax=False, batch=False)
        batchDice = DiceLoss(to_onehot_y=False, softmax=False, batch=True)
        mydice = DiceLoss(to_onehot_y=False, softmax=False, batch=False, my_dice=True)
        dicescore = dice(pred, gt_one_hot).cpu().detach().numpy().item(0)
        mydicescore = mydice(pred, gt_one_hot).cpu().detach().numpy().item(0)
        batchdicescore = batchDice(pred, gt_one_hot).cpu().detach().numpy().item(0)
        dice_scores.append(dicescore)
        mydice_scores.append(mydicescore)
        batchdice_scores.append(batchdicescore)

        # print(iter, "pred_class:", pred_class, "dice:", dicescore, "mydice:", mydicescore, "batchdice: ", batchdicescore, "avg_batch_misses:", avg_batch_miss,
        #       "sample_misses:", sample_miss)

    print("patch", patch, "pred_class:", pred_class, "dice:", np.array(dice_scores).mean(),
          "mydice:", np.array(mydice_scores).mean(), "batchdice:", np.array(batchdice_scores).mean(),
          "avg_batch_misses", np.array(avg_missing_classes_of_samples).mean(), "sample_misses", np.array(missing_classes_of_samples).mean())



if __name__ == "__main__":
    monai.utils.set_determinism(seed=0, additional_settings=None)

    eval_loss_functions(DATA_FULL, None, 1)
    eval_loss_functions(DATA_FULL, [192, 160, 56], 2)
    eval_loss_functions(DATA_FULL, [96, 80, 48], 8)
    eval_loss_functions(DATA_FULL, [8, 8, 8], 72)
