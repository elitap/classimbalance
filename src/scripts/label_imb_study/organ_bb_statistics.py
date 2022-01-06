import argparse
import SimpleITK as sitk
import numpy as np

import os

DATA_DIR = "./data/miccai/full_dataset_nifti/train"
LABEL_KEY = '_segmentation.nii.gz'

SPACING = [0.98000002, 0.98000002, 2.49962478]


def resample_img(itk_img, interpolation):

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolation)
    resampler.SetOutputSpacing(SPACING)

    imagescale = np.array(itk_img.GetSpacing()) / SPACING
    new_size = np.array(itk_img.GetSize()) * imagescale

    resampler.SetSize(new_size.round().astype(int).tolist())

    resampled_img = resampler.Execute(itk_img)
    resampled_img.SetOrigin([0, 0, 0])
    resampled_img.SetDirection(itk_img.GetDirection())
    resampled_img.SetSpacing(SPACING)

    return resampled_img


def iterate_dataset(dir, label_key):

    max_bb = [0, 0, 0]

    def update_max_bb(max_bb, curr_bb):
        for cnt, max in enumerate(max_bb):
            if max < curr_bb[cnt]:
                max_bb[cnt] = curr_bb[cnt]

    for f in os.listdir(dir):
        if label_key in f:

            print("Processing", f, "...")

            itk_label = sitk.ReadImage(os.path.join(dir, f))
            itk_label = resample_img(itk_label, sitk.sitkNearestNeighbor)

            on_chiasm = sitk.BinaryThreshold(itk_label, 1, 8)
            label_shapes_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shapes_filter.Execute(on_chiasm)
            label_bb = label_shapes_filter.GetBoundingBox(1)

            update_max_bb(max_bb, label_bb[3:])

    print("max bb: ", max_bb)




def main():
    iterate_dataset(DATA_DIR, LABEL_KEY)


if __name__ == "__main__":
    main()


