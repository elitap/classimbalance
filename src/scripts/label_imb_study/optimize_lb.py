from typing import Dict, Union, Tuple, List

import logging
import os
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib import rc

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from create_dataset import get_data
from util import config_parser, configure_logger
from config import Config

from contrib.transforms import RandCropByPosNeg
from monai.utils import set_determinism

LABEL_IDS = {
    0: "BG",
    1: "BS",
    2: "OC",
    3: "ON\_L",
    4: "ON\_R",
    5: "PG\_L",
    6: "PG\_R",
    7: "MA"
}


class Sampler:

    def __init__(self,
                 param: Config,
                 labels: List = None):
        self.param = param
        batches = self.param.batch_size
        self.param.batch_size = 1

        properties, self.dataLoader = get_data(param, mode="train")
        epoch_samples_per_image = self.param.num_epoch_batches // properties['numTraining']

        self.randCroper = None
        for transform in self.dataLoader.dataset.transform.transforms:
            if isinstance(transform, RandCropByPosNeg):
                self.randCroper : RandCropByPosNeg = transform

        self.randCroper.num_samples = epoch_samples_per_image * param.num_samples * batches

        self.labels = labels
        if self.labels is None:
            self.labels = list(map(int, properties['labels'].keys()))


    def label_imbalance(self,
                        label_map: Union[torch.Tensor, np.ndarray]) -> Tuple[List, List]:
        """
        function to calculate each's label percentage of the entire label map
        :param label_map: numpy ndarray of labelmap
        :return: dictionary of different labels with correspondent share
        """

        if torch.is_tensor(label_map):
            label_map = label_map.detach().cpu().numpy()

        # total = label_map.size

        unique, counts = np.unique(label_map, return_counts=True)
        label_ratio_dict = dict(zip(unique.astype(np.uint8), counts))
        label_vx = list()
        for label in self.labels:
            if label in label_ratio_dict.keys():
                label_vx.append(label_ratio_dict[label] / self.param.num_samples)
            else:
                label_vx.append(0)

        avg_label_vx = sum(label_vx)
        label_ratio = [lr/avg_label_vx for lr in label_vx]
        return label_ratio, label_vx


    def opt_func(self, x, y, z, r = -1):

        self.randCroper.spatial_size = [x, y, z]
        if r != -1:
            self.randCroper.pos_ratio = r

        label_ratios = list()
        label_vx = list()
        num_img_check = 0
        for cnt, batch in enumerate(self.dataLoader):
            ratios, vx = self.label_imbalance(batch['label'])
            label_ratios.append(ratios)
            label_vx.append(vx)
            num_img_check = cnt

        l = np.std(np.array(label_ratios))
        print("current vals:", x, y, z, r, l, num_img_check)

        return l, (x, y, z, r), np.mean(np.array(label_ratios), axis=0)


def optimize(param: Config) -> List:
    sampler = Sampler(param, labels=param['imb_labels'])

    log = logging.getLogger()

    min = 1.0
    opt = list()
    ratios = list()

    for z in range(1, 2, 1):
        for x in range(64, 257, 16):
            for y in range(64, 257, 16):
                for r in np.linspace(0.1, 0.8, 8):
                    l, opt_l, ratios_l = sampler.opt_func(x, y, z, r)
                    if l < min:
                        min = l
                        opt = opt_l
                        ratios = ratios_l
                        log.info(f"found new min: {l}, {opt_l}, {ratios}")

    log.info(f"Opt: {opt} for labels: {param['imb_labels']} with a std of: {l} foreground labels: {param.labels} "
             f"ratios {ratios}")



def plot_label_ratios(res_file, ratios: np.ndarray, title: str = None):
    plt.rcParams.update({'font.size': 18})

    rc('text', usetex=True)
    fig, axs = plt.subplots(1, 1, sharey=True, figsize=(6, 3))
    # fig.text(0.05, 0.5, 'Label ratio', va='center', rotation='vertical')

    plt.subplots_adjust(wspace=0.1)
    #fig.tight_layout(w_pad=.2)
    axs.set_yscale('log')
    axs.set_ylim([1e-6, 1e0])
    #axs.set_yticks([1e-6, 1e-4, 1e-2, 1e0])

    colors = list(mcolors.TABLEAU_COLORS.values())
    axs.grid(axis='y', linestyle='--')
    axs.bar(range(ratios.size), ratios, color=colors[:ratios.size])
    axs.set_xticks(range(ratios.size))
    axs.set_xticklabels(list(LABEL_IDS.values()), rotation=90)
    #ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    if title is not None:
        axs.title.set_text(title)

    plt.savefig(res_file, bbox_inches='tight', dpi=100)


def measure(param: Config) -> List:
    sampler = Sampler(param, labels=param['imb_labels'])

    log = logging.getLogger()

    x, y, z = param.patch_size

    l, opt_l, ratios_l = sampler.opt_func(x, y, z)

    log.info(f"std of: {l} foreground labels: {param.labels} ratios {ratios_l}")

    plot_label_ratios(f"./data/plots/large_ps.pdf", ratios_l, )
    # plot_label_ratios(f"./data/plots/fullvol_ps_{l}.pdf", ratios_l, f"Label ratios (full volume)")


def increase_num_filehandles():
    # TODO investigate has todo with to Cached dataset check how many
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

    #import torch.multiprocessing
    #torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--default_config',
                        default='./config/default.ini'
                        )
    # allows foregound sampling for specific labels only, is using all if not set
    parser.add_argument('-l', '--imb_labels',
                        nargs='+',
                        type=int,
                        default=None,
                        required=False)
    parser.add_argument('--experiment_config',
                        required=False,
                        help='allows to overwrite all default param for an experiment'
                        )
    parser.add_argument('--device',
                        required=False,
                        choices=[0, 1, 2, 3],
                        type=int
                        )

    increase_num_filehandles()
    args = vars(parser.parse_args())
    param = config_parser(args, generate_paths=True)

    if param.determinism:
        set_determinism(seed=param.determinism_seed)
        if param.local_debug_rank == 0:
            print("Using deterministic training.")

    if not os.path.exists(param.model_dir):
        os.makedirs(param.model_dir)

    configure_logger(param.train_log, local_debug_rank=0)

    # fix to use all labels
    if 'imb_labels' not in param.keys():
        param['imb_labels'] = None

    #measure(param)

    ratios_small = np.array([9.80502550e-01, 4.63969998e-03, 9.02099609e-05, 8.25005425e-05, 7.28434245e-05, 3.74850532e-03, 4.14233398e-03, 6.72135688e-03])
    plot_label_ratios(f"./data/plots/small_ps.pdf", ratios_small)

    ratios_large = np.array([9.87633591e-01, 3.63019206e-03, 6.02492560e-05, 6.87662760e-05, 6.25627790e-05, 2.25030227e-03, 2.22780180e-03, 4.06653413e-03])
    plot_label_ratios(f"./data/plots/large_ps.pdf", ratios_large)
    #
    ratios_full = np.array([9.98349192e-01, 3.25376987e-04, 6.34431839e-06, 8.39114189e-06, 7.53879547e-06, 3.32211256e-04, 3.42189074e-04, 6.28756285e-04])
    plot_label_ratios(f"./data/plots/full_ps.pdf", ratios_full)
    #
    ratios_min = np.array([8.40925347e-01, 3.70588108e-02, 1.02213542e-04, 1.61024306e-04, 1.94444444e-04, 3.13012153e-02, 3.42521701e-02, 5.60047743e-02])
    plot_label_ratios(f"./data/plots/min_ps.pdf", ratios_min)

    ratios_full_2d = np.array([9.97768894e-01, 4.91378128e-04, 1.44845696e-05, 9.22397901e-06, 9.96230751e-06, 3.63662679e-04, 3.61345148e-04, 9.81048871e-04])
    plot_label_ratios(f"./data/plots/large_ps_2d.pdf", ratios_full_2d)

    ratios_small_2d = np.array([9.78904724e-01, 3.94168837e-03, 4.35257523e-05, 3.74276620e-05, 3.75289352e-05, 4.17630208e-03, 4.20755208e-03, 8.65125145e-03])
    plot_label_ratios(f"./data/plots/small_ps_2d.pdf", ratios_small_2d)