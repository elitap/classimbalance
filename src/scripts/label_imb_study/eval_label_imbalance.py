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

from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    BorderPadd,
    RandSpatialCropd,
    AsChannelFirstd,
    RandAffined,
    Spacingd,
    ToTensord,
    ToNumpyd
)
from monai.utils import MetricReduction
from src.contrib.sampler import RandomRepeatingSampler

DATA = "./data/miccai/dataset_nifti/train"

DEFAULT_KEYS = {'img': "volume", 'seg': "segmentation", 'mask': "foreground"}

PATCH = [16, 16, 16]
BATCH = 512

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
                      key: dict = DEFAULT_KEYS,
                      spacing: tuple = None) -> DataLoader:
    data = os.path.abspath(data)
    label = sorted(glob(os.path.join(data, '*' + key["seg"] + '.nii.gz')))

    files = [{"img": v, "seg": l, "mask": m} for v, l, m in zip(volume, label, mask)]

    transform = [
        LoadNiftid(keys=["img", "seg", "mask"]),
        AddChanneld(keys=["img", "seg", "mask"]),
        AsChannelFirstd(keys=["mask"], channel_dim=0),
    ]

    if spacing is not None:
        transform.append(
            Spacingd(keys=['img', 'seg', "mask"], pixdim=spacing,
                     mode=('bilinear', 'nearest', 'nearest')),
        )

    transform_1 = [
        BorderPadd(keys=["img", "seg", "mask"], spatial_border=PATCH[0]),
        ToNumpyd(keys=["img", "seg", "mask"]),
        RandCropByPosNegLabeld(
            keys=["img", "seg", "mask"], label_key="mask",
            spatial_size=PATCH, pos=1, neg=0,
            num_samples=BATCH
        )
    ]

    transform += transform_1
    transforms = Compose(transform)

    ds = Dataset(data=files, transform=transforms)

    # sample repeat * batch patches from an image. (Note samples are different
    # for the each of the nth repetition even though randomness is turned off)
    repeat = 2
    dl = DataLoader(ds,
                    sampler=RandomRepeatingSampler(ds, repeat=repeat),
                    num_workers=8,
                    pin_memory=False) # do not use cuda for sampling test

    return dl


def label_imbalance(label_map: Union[torch.Tensor, np.ndarray]) -> Tuple[List, List]:
    """
    function to calculate each's label percentage of the entire label map
    :param label_map: numpy ndarray of labelmap
    :return: dictionary of different labels with correspondent share
    """

    if torch.is_tensor(label_map):
        label_map = label_map.detach().cpu().numpy()

    total = label_map.size

    unique, counts = np.unique(label_map, return_counts=True)
    label_ratio_dict = dict(zip(unique.astype(np.uint8), counts))
    label_vx = list()
    label_ratio = list()
    for label in range(len(LABEL_IDS.keys())):
        if label in label_ratio_dict.keys():
            label_ratio.append(label_ratio_dict[label]/total)
            label_vx.append(label_ratio_dict[label]/BATCH)
        else:
            label_ratio.append(0)
            label_vx.append(0)

    return label_ratio, label_vx


def eval_label_imbalence(data: str,
                         keys: dict = DEFAULT_KEYS,
                         spacing: tuple = None):

    loader = create_datalaoder(data, keys, spacing)

    label_ratios = list()
    label_vx = list()
    for cnt, data in enumerate(loader):
        print(data['seg_meta_dict']['filename_or_obj'][0], data['seg'].shape)
        ratios, vx = label_imbalance(data['seg'])
        label_ratios.append(ratios)
        label_vx.append(vx)
        #if cnt == 1:
        #    break
    return np.mean(np.array(label_ratios), axis=0)


def plot_label_ratios(ax, ratios: np.ndarray, title: str):

    colors = list(mcolors.TABLEAU_COLORS.values())
    ax.grid(axis='y', linestyle='--')
    ax.bar(range(ratios.size), ratios, color=colors[:ratios.size])
    ax.set_xticks(range(ratios.size))
    ax.set_xticklabels(list(LABEL_IDS.values()), rotation=90)
    #ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.title.set_text(title)


if __name__ == "__main__":
    monai.utils.set_determinism(seed=0, additional_settings=None)

    #rc('text', usetex=True)
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9, 4))
    fig.text(0.05, 0.5, 'Label ratio', va='center', rotation='vertical')

    plt.subplots_adjust(wspace=0.1)
    #fig.tight_layout(w_pad=.2)
    plt.yscale('log')

    lab_r = eval_label_imbalence(data=DATA, keys=DEFAULT_KEYS)
    plot_label_ratios(axs[0], lab_r, 'Original spacing')
    lab_r = eval_label_imbalence(data=DATA, keys=DEFAULT_KEYS, spacing=(1.1, 1.1, 1.1))
    plot_label_ratios(axs[1], lab_r, 'Isotrop spacing (1.1mm)')
    lab_r = eval_label_imbalence(data=DATA, keys=DEFAULT_KEYS, spacing=(2.2, 2.2, 2.2))
    plot_label_ratios(axs[2], lab_r, 'Isotrop spacing (2.2mm)')

    plt.savefig('./data/plots/label_imbalance_test.pdf', bbox_inches='tight', dpi=100)
    plt.show()
