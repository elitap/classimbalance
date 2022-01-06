import SimpleITK as sitk

import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.metrics import Metric
from monai.data.nifti_writer import write_nifti
from monai.engines import SupervisedEvaluator
from monai.engines.utils import IterationEvents, default_prepare_batch
from monai.inferers import Inferer
from monai.networks import eval_mode
from monai.transforms import AsDiscrete, Transform, KeepLargestConnectedComponent
from torch.utils.data import DataLoader

from transforms import recovery_prediction

from scripts.label_imb_study.save_prob_maps import save_prob


class DynUNetInferrer(SupervisedEvaluator):
    """
    This class inherits from SupervisedEvaluator in MONAI, and is used with DynUNet
    on Decathlon datasets.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be
            torch.DataLoader.
        network: use the network to run model forward.
        output_dir: the path to save inferred outputs.
        n_classes: the number of classes (output channels) for the task.
        epoch_length: number of iterations for one epoch, default to
            `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        post_transform: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        tta_val: whether to do the 8 flips (8 = 2 ** 3, where 3 represents the three dimensions)
            test time augmentation, default is False.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: DataLoader,
        network: torch.nn.Module,
        output_dir: str,
        n_classes: Union[str, int],
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        post_transform: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        tta_val: bool = False,
        cc_postproc_labels: Optional[Sequence[int]] = None,
        swap_symmetric: bool = False,
        save_prob_maps: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            post_transform=post_transform,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
            amp=amp,
        )

        if not isinstance(n_classes, int):
            n_classes = int(n_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=n_classes)
        self.output_dir = output_dir
        self.tta_val = tta_val
        self.n_classes = n_classes
        self.cc_postproc_labels = cc_postproc_labels
        self.swap_symmetric = swap_symmetric
        self.save_prob_maps = save_prob_maps

        if save_prob_maps:
            self.output_dir = os.path.join(self.output_dir, "prob_maps")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def _iteration(
        self, engine: Engine, batchdata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below item in a dictionary:
            - PRED: prediction result of model.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, label = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, label, args, kwargs = batch

        def _compute_pred():
            ct = 1.0
            pred = self.inferer(inputs, self.network, *args, **kwargs).cpu()
            pred = nn.functional.softmax(pred, dim=1)
            if not self.tta_val:
                return pred
            else:
                raise ValueError("tta option not supported any more")
                # for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                #     flip_inputs = torch.flip(inputs, dims=dims)
                #     flip_pred = torch.flip(
                #         self.inferer(flip_inputs, self.network).cpu(), dims=dims
                #     )
                #     flip_pred = nn.functional.softmax(flip_pred, dim=1)
                #     del flip_inputs
                #     pred += flip_pred
                #     del flip_pred
                #     ct += 1
                # return pred / ct

        # execute forward computation
        with eval_mode(self.network):
            if self.amp:
                with torch.cuda.amp.autocast():
                    predictions = _compute_pred()
            else:
                predictions = _compute_pred()

        filename = batchdata["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        affine = batchdata["image_meta_dict"]["affine"].numpy()[0]

        if self.save_prob_maps:
            print(
                "save {} probability map with shape: {}".format(
                    filename, predictions.shape
                )
            )

            # squeeze removes all dimensions of 1 so in this case the batch dim and the channel dim of the label
            save_prob(predictions=predictions.numpy().squeeze(),
                      label=label.cpu().numpy().squeeze(),
                      full_file_name=os.path.join(self.output_dir, filename),
                      n_classes=self.n_classes,
                      affine=affine)
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            return {"pred": predictions.numpy()}

        predictions = self.post_pred(predictions)
        if self.cc_postproc_labels is not None:
            cca = KeepLargestConnectedComponent(applied_labels=self.cc_postproc_labels)
            predictions = cca(predictions)

        resample_flag = batchdata["resample_flag"]

        anisotrophy_flag = batchdata["anisotrophy_flag"]
        crop_shape = batchdata["crop_shape"][0].tolist()
        original_shape = batchdata["original_shape"][0].tolist()

        # unique_before = np.unique(np.argmax(predictions.numpy()[0], axis=0))


        if resample_flag:
            # convert the prediction back to the original (after cropped) shape
            predictions = recovery_prediction(
                predictions.numpy()[0], [self.n_classes, *crop_shape], anisotrophy_flag
            )
        else:
            predictions = predictions.numpy()

        predictions = predictions[0]
        predictions = np.argmax(predictions, axis=0)

        # needs to be done if the nnunet dl is used but ugly and bad workaround and
        # should be removed in any case this gets ever published TODO
        if self.swap_symmetric:
            def swaper(npimg, left, right, tmp_max_label=100):
                npimg[npimg == left] = tmp_max_label
                npimg[npimg == right] = left
                npimg[npimg == tmp_max_label] = right

            swaper(predictions, 3, 4)
            swaper(predictions, 5, 6)

        # pad the prediction back to the original shape
        predictions_org = np.zeros(original_shape)
        box_start, box_end = batchdata["bbox"][0]
        h_start, w_start, d_start = box_start
        h_end, w_end, d_end = box_end
        predictions_org[h_start:h_end, w_start:w_end, d_start:d_end] = predictions
        del predictions


        print(
            "save {} with shape: {}".format(
                filename, predictions_org.shape
            )
        )
        write_nifti(
            data=predictions_org,
            file_name=os.path.join(self.output_dir, filename),
            affine=affine,
            resample=False,
            output_dtype=np.uint8,
        )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        return {"pred": predictions_org}
