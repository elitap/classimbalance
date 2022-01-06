from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import warnings

from monai.config import KeysCollection
from monai.transforms.croppad.array import (
    SpatialCrop,
)

from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utils import (
    generate_pos_neg_label_crop_centers,
    map_binary_to_indices,
)
from monai.utils import ImageMetaKey as Key
from monai.utils import fall_back_tuple

class RandCropByPosNeg(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding meta data.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        labels: if given only consider labels in list as foreground
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
            default is `meta_dict`, the meta data is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_missing_keys: don't raise exception if key is missing.
        batch_size: used to force foreground or background sampling, if not set batch_size is assumed to be num_samples

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        label_key: str,
        labels: Optional[List] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
        probabilistic_pos_neg: bool = True,
        batch_size: int = 0,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.labels = labels
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.meta_key_postfix = meta_key_postfix
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.probabilistic_pos_neg = probabilistic_pos_neg
        if batch_size <= 0:
            self.batch_size = num_samples
        else:
            self.batch_size = batch_size
        self.batch_idx = 0

    def generate_pos_neg_label_crop_centers(
            self,
            spatial_size: Union[Sequence[int], int],
            num_samples: int,
            pos_ratio: float,
            label_spatial_shape: Sequence[int],
            fg_indices: np.ndarray,
            bg_indices: np.ndarray,
            rand_state: np.random.RandomState = np.random,
    ) -> List[List[np.ndarray]]:
        """
        Generate valid sample locations based on the label with option for specifying foreground ratio
        Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

        Args:
            spatial_size: spatial size of the ROIs to be sampled.
            num_samples: total sample centers to be generated.
            pos_ratio: ratio of total locations generated that have center being foreground.
            label_spatial_shape: spatial shape of the original label data to unravel selected centers.
            fg_indices: pre-computed foreground indices in 1 dimension.
            bg_indices: pre-computed background indices in 1 dimension.
            rand_state: numpy randomState object to align with other modules.

        Raises:
            ValueError: When the proposed roi is larger than the image.
            ValueError: When the foreground and background indices lengths are 0.

        """
        spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
        if not (np.subtract(label_spatial_shape, spatial_size) >= 0).all():
            raise ValueError("The proposed roi is larger than the image.")

        # Select subregion to assure valid roi
        valid_start = np.floor_divide(spatial_size, 2)
        # add 1 for random
        valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
        # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
        # from being too high
        for i in range(len(valid_start)):  # need this because np.random.randint does not work with same start and end
            if valid_start[i] == valid_end[i]:
                valid_end[i] += 1

        def _correct_centers(
                center_ori: List[np.ndarray], valid_start: np.ndarray, valid_end: np.ndarray
        ) -> List[np.ndarray]:
            for i, c in enumerate(center_ori):
                center_i = c
                if c < valid_start[i]:
                    center_i = valid_start[i]
                if c >= valid_end[i]:
                    center_i = valid_end[i] - 1
                center_ori[i] = center_i
            return center_ori

        centers = []
        fg_indices, bg_indices = np.asarray(fg_indices), np.asarray(bg_indices)
        if fg_indices.size == 0 and bg_indices.size == 0:
            raise ValueError("No sampling location available.")

        if fg_indices.size == 0 or bg_indices.size == 0:
            warnings.warn(
                f"N foreground {len(fg_indices)}, N  background {len(bg_indices)}, "
                "unable to generate class balanced samples."
            )
            pos_ratio = 0 if fg_indices.size == 0 else 1

        for _ in range(num_samples):
            if self.probabilistic_pos_neg:
                indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
            else:
                indices_to_use = fg_indices if self.batch_idx < round(self.batch_size * pos_ratio) else bg_indices
            random_int = rand_state.randint(len(indices_to_use))
            center = np.unravel_index(indices_to_use[random_int], label_spatial_shape)
            # shift center to range of valid centers
            center_ori = list(center)
            centers.append(_correct_centers(center_ori, valid_start, valid_end))

            self.batch_idx += 1
            if self.batch_idx == self.batch_size:
                self.batch_idx = 0

        return centers

    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = self.generate_pos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_, bg_indices_, self.R
        )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> List[Dict[Hashable, np.ndarray]]:
        d = dict(data)
        label = d[self.label_key].copy()

        if self.labels:
            unique_labels = np.unique(label)
            for label_id in unique_labels:
                if label_id not in self.labels:
                    label[label == label_id] = 0

        image = d[self.image_key] if self.image_key else None
        fg_indices = d.get(self.fg_indices_key) if self.fg_indices_key is not None else None
        bg_indices = d.get(self.bg_indices_key) if self.bg_indices_key is not None else None

        self.randomize(label, fg_indices, bg_indices, image)
        if not isinstance(self.spatial_size, tuple):
            raise AssertionError
        if self.centers is None:
            raise AssertionError
        results: List[Dict[Hashable, np.ndarray]] = [{} for _ in range(self.num_samples)]

        for i, center in enumerate(self.centers):
            for key in self.key_iterator(d):
                img = d[key]
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
                results[i][key] = cropper(img)
            # fill in the extra keys with unmodified data
            for key in set(data.keys()).difference(set(self.keys)):
                results[i][key] = data[key]
            # add `patch_index` to the meta data
            for key in self.key_iterator(d):
                meta_data_key = f"{key}_{self.meta_key_postfix}"
                if meta_data_key not in results[i]:
                    results[i][meta_data_key] = {}  # type: ignore
                results[i][meta_data_key][Key.PATCH_INDEX] = i

        return results