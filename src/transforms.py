import os.path
from typing import Dict, Hashable, Mapping

import SimpleITK as sitk
import torch
import numpy as np
import logging
from monai.transforms import (CastToTyped,
                              Compose, CropForegroundd, LoadImaged,
                              NormalizeIntensity, RandRotated, RandAffined,
                              RandFlipd, RandGaussianNoised, RandAdjustContrastd,
                              RandGaussianSmoothd, RandScaleIntensityd, ScaleIntensityd,
                              RandZoomd, SpatialCrop, SpatialPadd, ToTensord,
                              AddChanneld, CenterSpatialCropd)

from contrib.transforms import RandCropByPosNeg

from monai.transforms.utils import generate_spatial_bounding_box
from skimage.transform import resize

from monai.transforms.compose import MapTransform
from monai.config import IndexSelection, KeysCollection


class AutoRemoveLastDimD(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    """

    def __init__(
        self,
        keys: KeysCollection,
        dimension: int = 3,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_size: the spatial size of output data after padding.
                If its components have non-positive values, the corresponding size of input image will be used.
            method: {``"symmetric"``, ``"end"``}
                Pad image symmetric on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.

        """
        super().__init__(keys)
        self.dimension = dimension

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            if self.dimension == 2:
                d[key] = d[key].squeeze(-1)
        return d


def get_task_transforms(mode, param):
    if mode != "test":
        keys = ["image", "label"]
    else:
        keys = ["image"]
        if param.save_prob_maps:
            keys += ["label"]

    load_transforms = [
        LoadImaged(keys=keys),
        # cool but only works in nightly
        # EnsureChannelFirstd(keys=keys),
        AddChanneld(keys=keys)
    ]
    # 2. sampling
    sample_transforms = [
        PreprocessAnisotropic(
            keys=keys,
            clip_values=param.clip,
            norm=param.norm,
            pixdim=param.spacing,
            model_mode=mode,
            force_isotropy=param.force_isotropy,
            crop_foreground=param.crop_foreground,
            save_prob_maps=param.save_prob_maps
        ),
        # ScaleIntensityd(keys=["image"])
    ]
    # 3. spatial transforms
    if mode == "train":
        other_transforms = [
            SpatialPadd(keys=["image", "label"], spatial_size=param.crop_size),
            RandCropByPosNeg(
                keys=keys,
                label_key="label",
                labels=param.labels,
                spatial_size=param.crop_size,
                pos=param.pos_sample_num,
                neg=param.neg_sample_num,
                num_samples=param.num_samples,
                image_key="image",
                image_threshold=0,
                probabilistic_pos_neg=param.probabilistic_pos_neg,
                batch_size=param.num_samples * param.batch_size,
            ),
            AutoRemoveLastDimD(keys=["image", "label"], dimension=param.dimension),
            RandAffined(
                keys=keys,
                spatial_size=param.crop_size[:param.dimension],
                prob=0.2,
                rotate_range=[np.pi, 0, 0] if param.dimension == 2 else [np.pi, np.pi/6.0, np.pi/6.0],
                scale_range=[-0.3, 0.4],
                mode=["bilinear", 'nearest'],
                padding_mode=['zeros', 'zeros'],
                as_tensor_output=False
            ),
            CenterSpatialCropd(
                keys=keys,
                roi_size=param.patch_size[:param.dimension],
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=[0.7, 1.5]
            ),
            # AutoRemoveLastDimD(keys=["image", "label"], dimension=param.dimension),
            # RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            # RandGaussianSmoothd(
            #      keys=["image"],
            #      sigma_x=(0.5, 1.15),
            #      sigma_y=(0.5, 1.15),
            #      sigma_z=(0.5, 1.15),
            #      prob=0.15,
            # ),
            # RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            # RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
            # RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            # RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
            CastToTyped(keys=keys, dtype=(np.float32, np.uint8)),
            ToTensord(keys=keys),
        ]
    elif mode == "validation":
        other_transforms = [
            SpatialPadd(keys=keys, spatial_size=param.patch_size),
            RandCropByPosNeg(
                keys=keys,
                label_key="label",
                labels=None,
                spatial_size=param.patch_size,
                pos=param.pos_sample_num,
                neg=param.neg_sample_num,
                num_samples=param.num_samples,
                image_key="image",
                image_threshold=0,
                batch_size=param.num_samples*param.batch_size,
                probabilistic_pos_neg=param.probabilistic_pos_neg,
            ),
            AutoRemoveLastDimD(keys=keys, dimension=param.dimension),
            CastToTyped(keys=keys, dtype=(np.float32, np.uint8)),
            ToTensord(keys=keys),
        ]
    else:
        other_transforms = [
            CastToTyped(keys=keys, dtype=(np.float32)),
            ToTensord(keys=keys),
        ]

    all_transforms = load_transforms + sample_transforms + other_transforms
    return Compose(all_transforms)


def resample_image(image, shape, anisotrophy_flag):
    resized_channels = []
    if anisotrophy_flag:
        for image_c in image:
            resized_slices = []
            for i in range(image_c.shape[-1]):
                image_c_2d_slice = image_c[:, :, i]
                image_c_2d_slice = resize(
                    image_c_2d_slice,
                    shape[:-1],
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_slices.append(image_c_2d_slice)
            resized = np.stack(resized_slices, axis=-1)
            resized = resize(
                resized,
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    else:
        for image_c in image:
            resized = resize(
                image_c,
                shape,
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    resized = np.stack(resized_channels, axis=0)
    return resized


def resample_label(label, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label)
    if anisotrophy_flag:
        shape_2d = shape[:-1]
        depth = label.shape[-1]
        reshaped_2d = np.zeros((*shape_2d, depth), dtype=np.uint8)

        for class_ in range(1, int(n_class) + 1):
            for depth_ in range(depth):
                mask = label[0, :, :, depth_] == class_
                resized_2d = resize(
                    mask.astype(float),
                    shape_2d,
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_
        for class_ in range(1, int(n_class) + 1):
            mask = reshaped_2d == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_
    else:
        for class_ in range(1, int(n_class) + 1):
            mask = label[0] == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_

    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


def recovery_prediction(prediction, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = shape[0]
    if anisotrophy_flag:
        c, h, w = prediction.shape[:-1]
        d = shape[-1]
        reshaped_d = np.zeros((c, h, w, d), dtype=np.uint8)
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized_d = resize(
                mask.astype(float),
                (h, w, d),
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped_d[class_][resized_d >= 0.5] = 1

        for class_ in range(1, n_class):
            for depth_ in range(d):
                mask = reshaped_d[class_, :, :, depth_] == 1
                resized_hw = resize(
                    mask.astype(float),
                    shape[1:-1],
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[class_, :, :, depth_][resized_hw >= 0.5] = 1
    else:
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized = resize(
                mask.astype(float),
                shape[1:],
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[class_][resized >= 0.5] = 1

    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


class PreprocessAnisotropic(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py

    """

    def __init__(
        self,
        keys,
        clip_values,
        norm,
        pixdim,
        model_mode,
        force_isotropy=False,
        crop_foreground=True,
        save_prob_maps=False,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.low = clip_values[0]
        self.high = clip_values[1]
        self.ds_mean = norm[0]
        self.ds_std = norm[1]
        self.target_spacing = pixdim
        self.training = False
        self.crop_foreg = CropForegroundd(keys=["image", "label"], source_key="image")
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)
        if model_mode in ["train", "validation"]:
            self.training = True

        self.save_prob_maps = save_prob_maps
        self.force_isotropy = force_isotropy
        self.crop_foreground = crop_foreground
        self.logger = logging.getLogger()

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def __call__(self, data):
        # load data
        d = dict(data)
        image = d["image"]
        image_spacings = d["image_meta_dict"]["pixdim"][1:4].tolist()

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0

        if self.training:
            if self.crop_foreground:
                cropped_data = self.crop_foreg({"image": image, "label": label})
                #print("origshape:", image.shape, "croppedshape", cropped_data["image"].shape)
                image, label = cropped_data["image"], cropped_data["label"]
        else:
            d["original_shape"] = np.array(image.shape[1:])
            ndim = len(d["original_shape"])
            box_start = [0] * ndim
            box_end = image.shape[1:]
            if self.crop_foreground:
                box_start, box_end = generate_spatial_bounding_box(image)
                image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image)
                if self.save_prob_maps:
                    label = SpatialCrop(roi_start=box_start, roi_end=box_end)(label)


            # print("origsahpe:", d["original_shape"], "croppedshape:", image.shape, "bbstart:", box_start, "bbend:", box_end)

            d["bbox"] = np.vstack([box_start, box_end])
            d["crop_shape"] = np.array(image.shape[1:])

        original_shape = image.shape[1:]
        # calculate shape
        resample_flag = False
        anisotrophy_flag = False


        def write_debug(np_img, prefix = ""):
            itk_image = sitk.GetImageFromArray(np.transpose(image[0], (2,1,0)))
            itk_image.SetOrigin((0,0,0))
            itk_image.SetSpacing(tuple(self.target_spacing))
            filename = os.path.split(d["image_meta_dict"]["filename_or_obj"])[1]
            sitk.WriteImage(itk_image, f"tmp/{prefix}_{filename}")

            logging.getLogger().error(f"{filename}, {anisotrophy_flag}")


        if self.target_spacing != image_spacings:
            # resample
            resample_flag = True
            resample_shape = self.calculate_new_shape(image_spacings, original_shape)
            anisotrophy_flag = self.check_anisotrophy(image_spacings) if not self.force_isotropy else False
            image = resample_image(image, resample_shape, anisotrophy_flag)

            if self.training or self.save_prob_maps:
                label = resample_label(label, resample_shape, anisotrophy_flag)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag

        # clip image for CT dataset
        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.ds_mean) / self.ds_std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image

        if "label" in self.keys:
            d["label"] = label

        return d
