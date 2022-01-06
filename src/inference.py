import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.distributed as dist
from monai.handlers import CheckpointLoader, MeanDice, StatsHandler
from monai.inferers import SlidingWindowInferer
from contrib.inferer import SlidingWindowInferer2
from monai.data import load_decathlon_properties
from torch.nn.parallel import DistributedDataParallel

from create_dataset import get_data
from create_network import get_network
from inferrer import DynUNetInferrer
from util import get_checkpoint_file, configure_logger, tool_exists
from config import config_parser

from scripts.eval.collect_results import evaluate
from scripts.eval.evaluate_results import evaluate_resultfile, evaluate_resultfile_organwise


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def inference(param):

    if param.multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{param.local_debug_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cuda:{param.device}")

    properties, test_loader = get_data(param, mode="test")

    if param.network == 'dynunet':
        net = get_network(properties, param)
        net = net.to(device)

    if param.multi_gpu:
        net = DistributedDataParallel(
            module=net, device_ids=[device], find_unused_parameters=True
        )

    save_dict = {'network': net}
    net.eval()

    ckpt_file, iteration = get_checkpoint_file(param)
    if ckpt_file is None:
        raise ValueError("No checkpoint found Validation does not make sense")

    ckpt_name = str(iteration)
    if iteration == -1:
        ckpt_name = "best"
    output_dir = os.path.join(param.model_dir, "out", ckpt_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if param.multi_gpu:
        # check out the map_param of the CheckpointLoader
        raise ValueError("Multi_gpu setting does not support checkpoint loading yet")

    val_handler = [CheckpointLoader(load_path=ckpt_file, load_dict=save_dict),
                   StatsHandler(output_transform=lambda x: None),
                   ]



    inferrer = DynUNetInferrer(
        device=device,
        val_data_loader=test_loader,
        network=net,
        output_dir=output_dir,
        n_classes=len(properties["labels"]),
        inferer=SlidingWindowInferer2(
            roi_size=param.patch_size,
            sw_batch_size=param.inference_slw_batch_size,
            overlap=param.inference_slw_overlap,
            mode=param.inference_slw_mode,
            device=torch.device('cpu'),
            dimension=param.dimension,
        ),
        # key_val_metric={
        #     "val_mean_dice": MeanDice(
        #         output_transform=lambda x: (x["pred"], x["label"].to(torch.device('cpu'))),
        #     )
        # },
        val_handlers=val_handler,
        amp=param.amp,
        cc_postproc_labels=list(map(int, properties['labels'].keys()))[1:],
        swap_symmetric=False,
        save_prob_maps=param.save_prob_maps
    )

    logfile = os.path.join(param.model_dir, "inference.log")
    configure_logger(logfile, param.local_debug_rank)

    inferrer.run()


def eval(param):

    properties = load_decathlon_properties(param.dataset_desc, ['labels'])

    resultfile = os.path.join(param.model_dir, 'out', 'results.csv')
    experiment = os.path.split(param.model_dir)[0]
    model = param.run
    threads = param.num_workers

    evaluate(gt_base_path=param.data_root,
             result_base_path=experiment,
             result_file=resultfile,
             model=model,
             special_checkpoint=None,
             result_dir='out',
             threads=threads,
             num_classes=len(properties['labels']),
             with_sd=False,
             checkpoint_filter='')
    evaluate_resultfile(resultfile, len(properties['labels']))
    evaluate_resultfile_organwise(resultfile)




if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--default_config',
                        default='./config/default.yaml'
                        )
    parser.add_argument('--experiment_config',
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

    inference(param)
    if tool_exists("plastimatch"):
        eval(param)
