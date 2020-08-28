from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from functools import partial
from durbango import *
from finetune import main as ft_main
from pathlib import Path
import os
def get_ray_slug(cfg):
    strang = ''
    for k,v in cfg.items():

        strang += f'{k}_{v}'
    for i in range(10000):
        test = f'rayruns/run_{i}'
        try:
            Path(test).mkdir(exist_ok=True,parents=True)
            break
        except Exception:
            continue

    return os.path.expanduser(test)


def ray_main(args, config):

    for k,v in config.items():
        #assert hasattr(args, k), k
        setattr(args, k, v)
    args.n_train = 64
    args.output_dir = get_ray_slug(config)
    args.num_train_epochs = 3
    ft_main(args)


def tune_helsinki_(args, num_samples=4, num_epochs=3):

    search_space = {
        "learning_rate": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
        "gradient_accumulation_steps": tune.choice([1, 8, 32, 128, 256]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.4]),
    }
    scheduler = ASHAScheduler(
        metric="val_avg_bleu",
        mode="min",
        max_t=3,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=list(search_space.keys()),
        metric_columns=["val_avg_loss", "val_avg_bleu", "global_step"])
    tune.run(
        partial(
            ray_main,
            args,
            ),
        resources_per_trial={"cpu": 0, "gpu": 1},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_helsinki_asha")

args = pickle_load('last_cl_args.pkl')
#ray_main(args, {})
tune_helsinki_(args)
