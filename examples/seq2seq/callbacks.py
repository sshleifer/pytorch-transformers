import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)


class Seq2SeqLoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True
    ) -> None:
        logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        return self._write_logs(trainer, pl_module, "test")


def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    """Saves the best model by validation ROUGE2 score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = FixedModelCheckpoint(
        filepath=os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        period=0,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback

from pytorch_lightning.utilities import rank_zero_warn
class FixedModelCheckpoint(ModelCheckpoint):

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # only run on main process
        if trainer.global_rank != 0:
            return

        metrics = pl_module.metrics['val'][-1]
        epoch = trainer.current_epoch
        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epoch_last_check is not None and (epoch - self.epoch_last_check) < self.period:
            # skipping in this term
            return

        self.epoch_last_check = epoch

        if self.save_last:
            filepath = os.path.join(self.dirpath, self.prefix + 'last.ckpt')
            self._save_model(filepath)

        filepath = self.format_checkpoint_name(epoch, metrics)
        version_cnt = 0
        while os.path.isfile(filepath):
            filepath = self.format_checkpoint_name(epoch, metrics, ver=version_cnt)
            # this epoch called before
            version_cnt += 1

        if self.save_top_k != -1:
            current = metrics.get(self.monitor)

            if not isinstance(current, torch.Tensor):
                rank_zero_warn(
                    f'The metric you returned {current} must be a `torch.Tensor` instance, checkpoint not saved'
                    f' HINT: what is the value of {self.monitor} in validation_epoch_end()?', RuntimeWarning
                )
                if current is not None:
                    current = torch.tensor(current)

            if current is None:
                rank_zero_warn(
                    f'Can save best model only with {self.monitor} available, skipping.', RuntimeWarning
                )
            elif self.check_monitor_top_k(current):
                self._do_check_save(filepath, current, epoch)
            elif self.verbose > 0:
                print(f'\nEpoch {epoch:05d}: {self.monitor}  was not in top {self.save_top_k}')

        else:
            if self.verbose > 0:
                print(f'\nEpoch {epoch:05d}: saving model to {filepath}')

            assert trainer.global_rank == 0, 'tried to make a checkpoint from non global_rank=0'
            self._save_model(filepath)


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )
