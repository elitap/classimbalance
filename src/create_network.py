import os

import torch
from monai.networks.nets.dynunet import DynUNet
#from contrib.nets_dynunet import DynUNet


def get_kernels_strides(patch_size, spacing, dimension):
    sizes, spacings = patch_size[:dimension], spacing[:dimension]
    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_network(properties, param):
    n_class = len(properties["labels"])
    in_channels = len(properties["modality"])

    kernels, strides = get_kernels_strides(param.patch_size, param.spacing, param.dimension)
    print(f'kernels: {kernels}, strides: {strides} used for training')
    # like nnunet use all except the last two steps
    deep_supervision = len(kernels) - 2

    weight = [0.5 ** i for i in range(deep_supervision)]
    weight_norm = [float(i) / sum(weight) for i in weight]
    print("deep supervistion weights will be: ", weight_norm)

    net = DynUNet(
        spatial_dims=param.dimension,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        # -1, as 0 is the output
        deep_supr_num=deep_supervision-1,
    )

    return net
