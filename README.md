# How to use Ray Tune with Lightning CLI? A dummy example on MNIST


This repository provides an example showing how to use Pytorch Lightning with Lightning CLI and Ray Tune.

## Usage

Exactly the same as without Ray Tune, so it should be something like:
```shell
python main.py fit -c config.yaml
```

## Code organization

- `main.py`: Here is where all Ray Tune is written. See below.
- `cli.py`: A subclass of the standard `LightningCLI` class. You should not have to modify anything here.
If you already have your custom implementation of CLI, just make this implementation be a subclass of yours.
- `data.py`: A dummy example of `LightningDataModule`.
- `model.py`: A dummy example of `LightningModule`.
- `config.yaml`: A dummy example of `yaml` config file.

## Main script

### Function `train_model`

If you use `LightningCLI` a standard way, there is nothing to change here.
Otherwise, e.g. if you want to do something before launching the training loop, write it down here.

### Function `tune_model`

Here is all the Ray Tune stuff that you can (should?) modify.
There are nice tutorials to show you what the different elements mean.

Unexhaustive list:
- [https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html]()
- [https://docs.ray.io/en/releases-1.11.0/tune/tutorials/tune-pytorch-lightning.html]()
