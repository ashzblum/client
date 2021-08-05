import wandb
from wandb.sweeps_engine.config import SweepConfig

sweep_config = {
        "program": "pcnntrain.py",
        "method": {"bayes": {"model": "tpe_multi"}},
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "epochs": {"min": 5, "max": 20, "distribution": "int_uniform"},
            "dropout": {"min": 0.25, "max": 1.5, "distribution": "uniform"},
            "batch_size": {"min": 50, "max": 590, "distribution": "int_uniform"},
            "channels_one": {"min": 8, "max": 56, "distribution": "int_uniform"},
            "channels_two": {"min": 16, "max": 126, "distribution": "int_uniform"},
            "learning_rate": {"min": 0, "max": 1, "distribution": "uniform"}
        }
    }

tuner = wandb.controller(SweepConfig(sweep_config))
tuner.run()
