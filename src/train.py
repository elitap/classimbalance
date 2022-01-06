from typing import Tuple, Union, Dict, Optional
import os
import numpy as np
import mlflow as mlf
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.distributed as dist
from ignite.engine import Events
from ignite.metrics import Average
from monai.handlers import (CheckpointSaver, LrScheduleHandler, MeanDice,
                            StatsHandler, ValidationHandler, CheckpointLoader, TensorBoardStatsHandler)
from monai.inferers import SimpleInferer

from monai.engines.utils import CommonKeys as Keys

from contrib.loss import DiceCELoss, DiceLoss
from torch.nn.modules.loss import _Loss
from monai.utils import set_determinism

from monai.engines.utils import default_prepare_batch
from torch.nn.parallel import DistributedDataParallel

from create_dataset import get_data
from create_network import get_network

from evaluator import DynUNetEvaluator
from trainer import DynUNetTrainer
from util import get_checkpoint_file, configure_logger, tool_exists
from config import config_parser
from inference import eval, inference


def train(param):

    # load hyper parameters
    if param.multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{param.local_debug_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cuda:{param.device}")

    properties, val_loader = get_data(param, mode="validation")
    _, train_loader = get_data(param, mode="train")
    prepare_patch_fn = default_prepare_batch

    # produce the network
    if param.network == 'dynunet':
        net = get_network(properties, param)
        net = net.to(device)
        # find difference between trainable and total num params
        pytorch_params = sum(p.numel() for p in net.parameters())
        print("num trainable params", pytorch_params)


    if param.multi_gpu:
        net = DistributedDataParallel(
            module=net, device_ids=[device], find_unused_parameters=True
        )

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=param.lr,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (1 - epoch / param.epochs) ** 0.9
    )

    # TODO find out how to checkpoint epoch
    save_dict = {'network': net, 'optimizer': optimizer}
    if param.lr_decay:
        save_dict['lr_scheduler'] = scheduler

    # produce evaluator
    val_handlers = [
        CheckpointSaver(save_dir=param.ckpt_dir,
                        save_dict=save_dict,
                        key_metric_name="val_mean_dice",
                        save_key_metric=True,
                        save_final=False,
                        epoch_level=True
                        ),
        StatsHandler(output_transform=lambda x: None,
                     global_epoch_transform=lambda x: trainer.state.epoch),
        TensorBoardStatsHandler(
            log_dir=os.path.join(param.tensorboard_log, "eval"),
            output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
            global_epoch_transform=lambda x: trainer.state.iteration,
        )
    ]

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        epoch_length=param.val_epoch_length,
        n_classes=len(properties["labels"]),
        prepare_batch=prepare_patch_fn,
        post_transform=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                output_transform=lambda x: (x["pred"], x["label"]),
            )
        },
        val_handlers=val_handlers,
        amp=param.amp
    )

    # produce trainer
    loss = get_loss(param.loss)

    train_handlers = [
        CheckpointSaver(save_dir=param.ckpt_dir,
                        save_dict=save_dict,
                        save_key_metric=False,
                        save_final=True,
                        epoch_level=True
                        ),
        ValidationHandler(validator=evaluator, interval=param.val_interval, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x[Keys.LOSS]),
        TensorBoardStatsHandler(log_dir=param.tensorboard_log,
                                output_transform=lambda x: x['loss'].data.tolist())
    ]

    if param.lr_decay:
        train_handlers += [LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)]

    ckpt_file, iteration = get_checkpoint_file(param)
    if ckpt_file is not None:
        if param.multi_gpu:
            # check out the map_param of the CheckpointLoader
            raise ValueError("Multi_gpu setting does not support checkpoint loading")
        train_handlers += [CheckpointLoader(load_path=ckpt_file, load_dict=save_dict)]

    trainer = DynUNetTrainer(
        device=device,
        max_epochs=param.epochs,
        epoch_length=param.train_epoch_length,
        prepare_batch=prepare_patch_fn,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        #post_transform=None,
        key_train_metric={
            "avg_loss": Average(
                output_transform=lambda x: x[Keys.LOSS],
            ),
        },
        train_handlers=train_handlers,
        amp=param.amp,
    )

    # todo find a way to checkpoint epochs as well
    if iteration > 0 and not param.multi_gpu:
        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = iteration // engine.state.epoch_length
            engine.state.iteration = iteration

    # log and run
    # todo probably report individual dice scores here
    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_metric(engine):
        mlf.log_metric("val_dice", engine.state.metrics['val_mean_dice'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metric(engine):
        mlf.log_metric("loss", engine.state.metrics['avg_loss'])
        if param.lr_decay:
            mlf.log_metric("lr", scheduler.get_last_lr()[0])

    configure_logger(param.train_log, param.local_debug_rank)

    trainer.run()


def get_loss(loss: str) -> _Loss:
    avail_losses = {
        "ce_dice":
            DiceCELoss(to_onehot_y=True, softmax=True, batch=False),
        "ce_mydice":
            DiceCELoss(to_onehot_y=True, softmax=True, batch=False, my_dice=True),
        "ce_mydice_nb":
            DiceCELoss(to_onehot_y=True, softmax=True, batch=False, my_dice=True, include_background=False),
        "ce_mydice_bd":
            DiceCELoss(to_onehot_y=True, softmax=True, batch=True, my_dice=True),
        "ce_mydice_bd_nb":
            DiceCELoss(to_onehot_y=True, softmax=True, batch=True, my_dice=True, include_background=False),
        "ce_dice_nb":
            DiceCELoss(to_onehot_y=True, softmax=True, batch=False, include_background=False),
        "ce_dice_bd_nb":
            DiceCELoss(to_onehot_y=True, softmax=True, batch=True, include_background=False),
        "mydice":
            DiceLoss(to_onehot_y=True, softmax=True, batch=False, my_dice=True),
        "dice_bd_nb":
            DiceLoss(to_onehot_y=True, softmax=True, batch=True, include_background=False),
        "redsum_ce_dice":
            DiceCELoss(to_onehot_y=True, softmax=True, batch=False, reduction='sum'),
        "dice":
            DiceLoss(to_onehot_y=True, softmax=True, batch=False),
    }
    return avail_losses[loss]


def log_environment(args):
    from pip._internal.operations.freeze import freeze
    with open("environment.txt", 'w') as out_file:
        for requirement in freeze(local_only=True):
            out_file.write(requirement+"\n")
    mlf.log_artifact("environment.txt")
    mlf.log_artifact(args['default_config'])
    if args["experiment_config"] is not None and os.path.exists(args["experiment_config"]):
        mlf.log_artifact(args['experiment_config'])


def log_param(param):

    mydict = param.asdict()

    for k in mydict:
        mlf.log_param(k, mydict[k])


def increase_num_filehandles():
    # TODO investigate has todo with to Cached dataset check maybe new Monai version
    # Only fails with smallres runs where the bs is larger

    # this works for 3d
    # on server ulimit -n 512000 to make it work
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

    # seems to be required for 2d
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":

    increase_num_filehandles()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--default_config',
                        default='./config/default.yaml'
                        )
    parser.add_argument('-e', '--experiment_config',
                        required=False,
                        help='allows to overwrite all default param for an experiment'
                        )
    parser.add_argument('--device',
                        required=False,
                        choices=[0, 1, 2, 3],
                        type=int
                        )

    args = vars(parser.parse_args())
    param = config_parser(args)

    if param.determinism:
        set_determinism(seed=param.determinism_seed)
        if param.local_debug_rank == 0:
            print("Using deterministic training.")

    mlf.set_tracking_uri(uri="./mlruns")
    mlf.set_experiment(experiment_name=param.experiment_name)

    with mlf.start_run(run_name=param.run):
        print(mlf.get_artifact_uri())
        mlf.log_artifacts("src", "src")
        log_environment(args)
        log_param(param)

        train(param)

        mlf.log_artifact(param.train_log)

    inference(param)
    if tool_exists("plastimatch"):
        eval(param)

