from typing import Optional
import numpy as np

from monai.data.nifti_writer import write_nifti



def save_prob(predictions: np.ndarray,
              label: np.ndarray,
              full_file_name: str,
              n_classes: int,
              affine: Optional[np.ndarray] = None):

    prob_map = np.zeros_like(label, dtype=np.float)

    for label_id in range(n_classes):
        # store the probability of the correct label, should be as high as possible
        prob_map[label == label_id] = predictions[label_id][label == label_id]

    write_nifti(
        data=prob_map,
        file_name=full_file_name,
        affine=affine,
        resample=False,
        output_dtype=np.float,
    )

    # write_nifti(
    #     data=label,
    #     file_name=full_file_name.replace('volume', 'label'),
    #     affine=affine,
    #     resample=False,
    #     output_dtype=np.uint8,
    # )





