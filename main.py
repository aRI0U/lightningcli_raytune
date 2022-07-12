from pprint import pprint

import pytorch_lightning as pl

import ray.tune as tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback


from cli import RayTuneCli


def train_model(config, cli, callbacks):
    cli.instantiate_and_run(config, callbacks)


def tune_model(cli):
    # hyperparameters to search
    hparams = {
        "model.init_args.lr": tune.loguniform(1e-4, 1e-1),
        "model.init_args.layer_1_size": tune.choice([32, 64, 128]),
        "model.init_args.layer_2_size": tune.choice([64, 128, 256])
    }

    # metrics to track: keys are displayed names and
    # values are corresponding labels defined in LightningModule
    metrics = {
        "loss": "ptl/val_loss",
        "accuracy": "ptl/val_accuracy"
    }

    # scheduler
    scheduler = ASHAScheduler(
        max_t=3,
        grace_period=1,
        reduction_factor=2
    )

    # progress reporter
    reporter = CLIReporter(
        parameter_columns={p: p.split('.')[-1] for p in hparams.keys()},
        metric_columns=list(metrics.keys())
    )

    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    resources_per_trial = {
        "cpu": 2,
        "gpu": 0
    }

    # main analysis
    trainable_function = tune.with_parameters(
        train_model,
        cli=cli,
        callbacks=callbacks
    )
    result = tune.run(
        trainable_function,
        resources_per_trial=resources_per_trial,
        config=hparams,
        scheduler=scheduler,
        progress_reporter=reporter,
        metric="loss",
        mode="min",
        num_samples=3
    )

    best_trial = result.get_best_trial("accuracy", "max", "all")
    print("Best hyperparameters found were:")
    pprint(best_trial.config)
    print("Corresponding metrics are:")
    pprint({metric: best_trial.last_result[metric] for metric in metrics.keys()})


if __name__ == "__main__":
    cli = RayTuneCli(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True
    )
    tune_model(cli)
