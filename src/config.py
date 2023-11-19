import os

import numpy as np
import yaml

from typing import Dict, TYPE_CHECKING, List, Tuple


def parse_options(param):
    # function to parse read parameters, sets "None" to None and "[]" to []

    for key in list(param.__annotations__.keys()):
        if param.__getattribute__(key) == "None":
            param.__setattr__(key, None)
        if param.__getattribute__(key) == "[]":
            param.__setattr__(key, [])


if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass()
class Config:
    experiment_name: str
    run: str
    data_root: str
    dataset_desc: str
    num_workers: int
    val_num_workers: int
    val_interval: int
    num_val_batches: int
    ds_cache_rate: float
    ds_cache_dir: str | None
    continue_training: bool
    checkpoint_file: str
    epochs: int
    num_epoch_batches: int
    amp: bool
    lr_decay: bool
    determinism: bool
    determinism_seed: int
    multi_gpu: bool
    local_debug_rank: int
    inference_slw_overlap: float
    inference_slw_batch_size: int
    inference_slw_mode: str
    device: int

    batch_size: int
    num_samples: int
    probabilistic_pos_neg: bool
    pos_sample_num: int
    neg_sample_num: int
    labels: List[str]
    loss: str
    lr: float
    network: str
    patch_size: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    force_isotropy: bool
    save_prob_maps: bool
    crop_foreground: bool
    clip: Tuple[int, int]
    norm: Tuple[float, float]
    dimension: int

    default_config: str
    model_dir: str
    ckpt_dir: str
    checkpoint_file: str
    train_log: str
    tensorboard_log: str
    train_epoch_length: int
    val_epoch_length: int
    crop_size: List[int]

    def update_parameters(self, new_param: Dict):
        """
        method to update current stored parameters with parameters given in new_param
        Args:
            new_param: Dictionary, holding attribute name as key and according value as value
        Returns: None
        """
        for key in new_param:
            try:
                setattr(self, key, new_param[key])
            except AttributeError:
                print("Could not update parameter " + key + "=" + str(new_param[key]) + ". No such attribute found.")

    def asdict(self) -> Dict:
        """
        method returns current parameters as dictionary
        Returns: Dictionary
        """
        mydict = {}

        for x in list(self.__annotations__.keys()):
            mydict[x] = str(self.__getattribute__(x))

        return mydict

    def print(self):
        """
        method prints current parameters to console
        Returns: None
        """
        for x in list(self.__annotations__.keys()):
            print(x + ": " + str(self.__getattribute__(x)))


def config_parser(args: Dict, generate_paths: bool = True) -> Config:
    """
    function to read parameters from file and create Config-Object
    Args:
        args: programm arguments, holding paths to configs
        generate_paths: flag to decide whether absolute paths are generated or not

    Returns: Config

    """

    if not os.path.exists(args['default_config']):
        raise ValueError("default config file must exist")

    with open(args['default_config']) as file:      # as it has been tested before if path exists, this should always be possible
        full_config = yaml.safe_load(file)

    full_config["default_config"] = args['default_config']

    if args["experiment_config"] is not None and os.path.exists(args["experiment_config"]):
        with open(args['experiment_config']) as experiment_file:
            new_parameters = yaml.safe_load(experiment_file)

            for key, value in new_parameters.items():
                if key not in full_config.keys():
                    print("Could not update parameter " + key + "=" + str(new_parameters[key]) + ". No such attribute found.")
                else:
                    full_config[key] = value

            # checked update would require default values in dataclass and pydantic and python dataclass have
            # different field definitions
            # param = Config(**full_config)
            # param.update_parameters(new_param=new_parameters)
            # full_config = param.asdict()

    if args["experiment_config"] is not None and not os.path.exists(args["experiment_config"]):
        raise ValueError(f"experiment_config but given file does not exist")



    # from read parameters calculate some more properties and add them to config dict
    if generate_paths:
        if not os.path.exists(full_config['data_root']):
            raise ValueError(f"dataset root {full_config['data_root']} does not exist")
        if not os.path.exists(full_config['dataset_desc']):
            raise ValueError(f"dataset json {full_config['dataset_desc']} does not exist")

        if full_config['inference_slw_mode'] not in ['gaussian', 'constant']:
            raise ValueError(f"unsupported sliding window mode {full_config['inference_slw_mode']}")

        full_config['model_dir'] = os.path.join("./models", full_config['experiment_name'], full_config['run'])
        full_config['ckpt_dir'] = os.path.join(full_config['model_dir'], "ckpt")
        full_config['checkpoint_file'] = os.path.join(full_config['ckpt_dir'], full_config['checkpoint_file'] if full_config['checkpoint_file'] is not None else 'invalid')
        full_config['train_log'] = os.path.join(full_config['model_dir'], "train.log")
        full_config['tensorboard_log'] = os.path.join(full_config['model_dir'], "log")

    else:   # empty strings as default values
        full_config['model_dir'] = ""
        full_config['ckpt_dir'] = ""
        full_config['checkpoint_file'] = ""
        full_config['train_log'] = ""
        full_config['tensorboard_log'] = ""

    full_config['train_epoch_length'] = full_config['num_epoch_batches']  # // (param.num_samples * param.batch_size)
    full_config['val_epoch_length'] = full_config['num_val_batches']  # // (param.num_samples * param.batch_size)
    full_config['crop_size'] = (np.array(full_config['patch_size']) * 1.4).astype(int).tolist()

    # create instance of Config out of read parameters
    param = Config(**full_config)

    if args['device'] is not None:
        param.device = args['device']

    param.default_config = args["default_config"]

    parse_options(param)    # parse "None" and "[]" to python values

    if param.dimension == 2:
        param.crop_size[-1] = 1

    if param.network == 'transunet' and param.dimension == 3:
        raise ValueError(f"transunet only supports 2d inputs")

    # print config for user info
    param.print()

    return param


if __name__ == "__main__":

    # for testing only

    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--default_config',
                        default='./config/default.yaml'
                        )
    parser.add_argument('--experiment_config',
                        default='./config/experiments/nnunet3d_caDice_ce.yaml',
                        #required=False,
                        help='allows to overwrite all default param for an experiment'
                        )
    parser.add_argument('--device',
                        required=False,
                        choices=[0, 1, 2, 3],
                        type=int
                        )

    my_args = vars(parser.parse_args())
    my_param = config_parser(my_args)
