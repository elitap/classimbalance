# Info

This is the code repository of the work **Tackling the Class Imbalance Problem of Deep Learning Based Head and Neck Organ Segmentation** from Elias Tappeiner, Martin Welk and Rainer Schubert.

The implementation is based on the [Monai DynUNet pipeline module](https://github.com/Project-MONAI/tutorials/), a reimplementation of the dynamic UNet used in the [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) of Isensee et. al and further adapted to follow the nnU-Net parameterization.

### Requirements

+ Python: [requirments.txt](requirements.txt) (the presented results are generated using the given [config files](config) and the python [environment](environment.txt))
+ Libs: [plastimatch](https://plastimatch.org/) (additional for the result evaluation)


# Code 

### Dataset preparation

1. Download the MICCAI Head and Neck segmentation challenge [dataset](http://www.imagenglab.com/newsite/pddca/)
2. Run ``python src/scripts/combine_dataset_label_files.py --datasetpath path/to/unpacked/datasetfiles/ --outdir data/pddca``
3. The dataset with combined labelmaps can now be found under [data/pddca](data/pddca)
4. In [config/data](config/data) a [segmentation decatlon](http://medicaldecathlon.com/) conform [json file](config/data/task_HaN.json) for the dataset is defined, which is used throughout the code to access the data


### Train
Simply run the training script using one of the given [config files](config/experiments) or use your own. Details about the available configuration options are found in the [default config file](config/default.yaml).

e.g. training the nnU-Net baseline:
```python
python src/inference.py --experiment_config config/experiments/nnunet3d_nnUDice_ce.yaml 
```

Tensorboard logs are written to the model directory of the experiment, the parameter configuration of all experiments is logged using mlflow and can be found in [mlruns](mlruns) after the training.


### Inference
1. Pretrained weights are given [here](https://drive.google.com/file/d/1PvQzxbLDM5gXdfiwQeCwI2ymfuZc4E-9/view?usp=sharing) (skip next point if you trained the model by your own)
2. Extract the pretrained experiment models to [models/cars22/](models/cars22/) (the baseline weights should then be found under models/cars22/nnunet3d_caDice_ce/ckpt/checkpoint_final_iteration=125000.pt)
3. e.g. to infer the baseline experiment run (change yaml file to run your own):
```python
python src/inference.py --experiment_config config/experiments/nnunet3d_nnUDice_ce.yaml 
```
