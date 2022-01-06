from typing import Dict, TYPE_CHECKING, List, Tuple

import yaml
import os
import re
import logging
import numpy as np

from config import Config

def configure_logger(logfile, local_debug_rank):
    logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Setup file handler
    fhandler = logging.FileHandler(logfile)
    fhandler.setLevel(logging.INFO)
    fhandler.setFormatter(formatter)
    # Configure stream handler for the cells
    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    chandler.setFormatter(formatter)
    # Add both handlers
    if local_debug_rank == 0:
        logger.addHandler(fhandler)
        logger.addHandler(chandler)
        logger.setLevel(logging.INFO)


def get_checkpoint_file(param: Config):
    if param.continue_training:
        if os.path.exists(param.checkpoint_file):
            return [param.checkpoint_file, -1]

    if os.path.exists(param.ckpt_dir):

        ckpts = list()
        ckpt_basenames = list()

        for ckp in os.listdir(param.ckpt_dir):
            re_groups = re.search(r'(.*)=(\d+).pt', ckp)
            if re_groups:
                ckpt_basenames.append(str(re_groups.group(1)))
                ckpts.append(int(re_groups.group(2)))

        if len(ckpts) == 0:
            return [None, -1]

        filename = ckpt_basenames[ckpts.index(max(ckpts))]+"="+str(max(ckpts))+".pt"

        return [os.path.join(param.ckpt_dir, filename), max(ckpts)]

    return [None, -1]


def tool_exists(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None