import argparse
import gc
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import git
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from rouge_score import rouge_scorer, scoring
from torch import nn
from torch.utils.data import DataLoader

from durbango import lmap, pickle_load, pickle_save
from lightning_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    BartConfig,
    BartForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.modeling_bart import invert_mask
from transformers.optimization import AdamW


try:
    from .utils import SummarizationDataset, flatten_list
    from .bart_distiller import init_student, copy_layers
except ImportError:
    from utils import SummarizationDataset, flatten_list
    from bart_distiller import init_student, copy_layers

logger = logging.getLogger(__name__)

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def save_git_info(folder_path: str):
    """
    Log commit info.
    """
    repo_infos = get_git_info()

    with open(os.path.join(folder_path, "git_log.json"), "w") as f:
        json.dump(repo_infos, f, indent=4)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }
    return repo_infos


def calculate_rouge(output_lns: List[str], reference_lns: List[str], all_stats=False) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    if all_stats:
        return expanded_rouge_df(result)
    else:
        return {k: v.mid.fmeasure for k, v in result.items()}


def dictify(rouge_obj) -> List:
    records = []
    for k, rouge_measurement in rouge_obj.items():
        if k == "rouge1":
            continue
        for k1 in ["low", "mid", "high"]:
            if k1 != "mid":
                continue
            v1 = getattr(rouge_measurement, k1)
            for k2 in ["precision", "recall", "fmeasure"]:
                records.append([k, k1, k2, getattr(v1, k2)])

    return records


def expanded_rouge_df(rouge_all) -> pd.DataFrame:
    return (
        pd.DataFrame(dictify(rouge_all), columns=["metric", "k1", "k2", "val"])
        .set_index(["metric", "k2"])["val"]
        .unstack("metric")
        .rename_axis(None)
    )


class SummarizationTrainer(BaseTransformer):
    mode = "language-modeling"
    loss_names = ["loss"]

    @property
    def is_t5(self):
        return self.model.config.model_type == "t5"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        save_git_info(self.hparams.output_dir)
        self.model: AutoModelWithLMHead
        self.metrics_save_path = Path(self.output_dir) / "metrics.pkl"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        self.step_count = 0

        if os.path.exists(self.metrics_save_path):
            self.metrics = pickle_load(self.metrics_save_path)
        else:
            self.metrics = {"train": [], "val": [], "test": []}

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            overwrite_cache=self.hparams.no_cache,
            tgt_suffix=self.hparams.tgt_suffix,
        )
        base_nobs = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_mtl,
            "test": self.hparams.test_mtl,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        self.n_obs = {k: v if v >= 0 else None for k, v in base_nobs.items()}
        self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_part(self.model.encoder)
        self.hparams.git_sha = get_git_info()["repo_sha"]
        pickle_save(self.hparams, self.hparams_save_path)
        # self.hparams.

    def freeze_embeds(self):
        freeze_part(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_part(d.embed_positions)
            freeze_part(d.embed_tokens)

    @property
    def metrics_df(self):
        return pd.DataFrame(self.metrics)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)

        loss = outputs[0]

        return (loss,)

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        rouges = {k: np.array([x[k] for x in outputs]).mean() for k in ROUGE_KEYS + ["gen_time"]}
        rouge: torch.FloatTensor = torch.tensor(rouges["rouge2"]).type_as(loss)
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)
        preds = flatten_list([x["preds"] for x in outputs])
        ret_dict = {"log": metrics, "preds": preds}
        ret_dict[f"{prefix}_loss"] = loss
        ret_dict[f"{prefix}_rouge"] = rouge
        return ret_dict

    def save_metrics(self, metrics, prefix) -> None:
        self.metrics[prefix].append(metrics)
        pickle_save(self.metrics, self.metrics_save_path)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _generative_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        cfg = self.config
        # {'max_length': cfg.max_length, 'min_length': cfg.min_length, 'num_beams': cfg.num_beams}

        t0 = time.time()
        generated_ids = self.model.generate(input_ids=source_ids, attention_mask=source_mask, use_cache=True,)
        gen_time = time.time() - t0
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(y)
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = calculate_rouge(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, summ_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_end(self, outputs):
        return self.validation_end(outputs, prefix="test")

    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+") as p_writer, open(output_test_targets_file, "w+") as t_writer:
            for output_batch in outputs:
                p_writer.writelines(s + "\n" for s in output_batch["preds"])
                t_writer.writelines(s + "\n" for s in output_batch["target"])
            p_writer.close()
            t_writer.close()

        return self.test_end(outputs)

    def validation_epoch_end(self, outputs):
        self.validation_end(outputs, "val")

    def get_dataset(self, type_path) -> SummarizationDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = SummarizationDataset.from_raw_data(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:

        dataset = self.get_dataset(type_path)
        sampler = None
        if self.hparams.sortish_sampler and type_path == "train":
            sampler = dataset.make_sortish_sampler(batch_size)
            shuffle = False
            assert self.hparams.gpus <= 1

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            # num_workers=4,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        grps = [[], [], []]

        def get_group(n):
            no_decay = any(nd in n for nd in ["bias", "LayerNorm.weight"])
            if self.hparams.freeze_decoder and "decoder" in n:
                return 2
            elif no_decay:
                return 1
            else:
                return 0

        for n, p in model.named_parameters():
            g = get_group(n)
            grps[g].append(p)

        optimizer_grouped_parameters = [
            {"params": grps[0], "weight_decay": self.hparams.weight_decay,},
            {"params": grps[1], "weight_decay": 0.0,},
            {"params": grps[2], "learning_rate": 0.0,},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        # Add BART specific options
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_mtl",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_mtl",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
        )
        parser.add_argument(
            "--no_cache", action="store_true",
        )
        parser.add_argument(
            "--freeze_encoder", action="store_true",
        )
        parser.add_argument(
            "--freeze_decoder", action="store_true",
        )
        parser.add_argument("--tgt_suffix", type=str, default="", required=False)
        parser.add_argument("--n_train", type=int, default=-1, required=False)
        parser.add_argument("--n_val", type=int, default=500, required=False)
        parser.add_argument("--n_test", type=int, default=-1, required=False)
        parser.add_argument("--sortish_sampler", action="store_true", default=False)

        return parser


def freeze_part(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def is_frozen(model):
    return not any(p.requires_grad for p in model.parameters())


def get_layers_to_copy(n_to_get, tot):
    all_layers = list(range(tot))
    if tot == 12:  # Alternating for special cases
        layers_to_copy = {  # maps # layers in student -> which teacher layers to copy
            6: [0, 2, 4, 7, 9, 11],
            1: [11],
            3: [0, 6, 11],
            2: [0, 11],
            4: [0, 4, 8, 11],
            9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
            12: all_layers,
        }
        return layers_to_copy[n_to_get]
    else:
        return all_layers[:n_to_get]


BART_LARGE_N_LAYERS = 12


class SummarizationDistiller(SummarizationTrainer):
    loss_names = ["loss", "ce_loss", "mlm_loss", "enc_mse_loss"]
    teacher_kwargs = {}

    def __init__(self, hparams):
        assert Path(hparams.data_dir).exists()

        d_layers_to_copy, student, student_cfg, teacher = self.pre_init(hparams)

        super().__init__(hparams, model=student, config=student_cfg)
        self.teacher = teacher
        freeze_part(self.teacher)

        self.freeze_stuff(d_layers_to_copy)

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.temperature = 2.0
        self.alpha_mlm = hparams.alpha_mlm
        self.alpha_ce = hparams.alpha_ce
        self.alpha_hid = hparams.alpha_hid
        # self.alpha_cos = hparams.alpha_cos
        self.alpha_encoder_loss = self.hparams.alpha_encoder_loss
        gc.collect()
        torch.cuda.empty_cache()

    def freeze_stuff(self, d_layers_to_copy):
        assert len(self.model.decoder.layers) == len(d_layers_to_copy)
        assert_all_frozen(self.teacher)
        assert_all_frozen(self.model.decoder.embed_tokens)
        assert_all_frozen(self.model.encoder.embed_tokens)
        if self.different_encoder:
            assert any(grad_status(self.model.encoder))
        else:
            freeze_part(self.model.encoder)
            del self.teacher.model.encoder
        if self.different_decoder:
            assert any(grad_status(self.model.decoder))
        else:
            freeze_part(self.model.decoder)  # TODO(SS): very suspicious

    def pre_init(self, hparams):
        # Dump empty student model at a path, then call from_pretrained on it
        teacher = BartForConditionalGeneration.from_pretrained(hparams.teacher, **self.teacher_kwargs).eval()
        student_updates = {
            "decoder_layers": hparams.student_decoder_layers,
            "encoder_layers": hparams.student_encoder_layers,
        }
        d_layers_to_copy = get_layers_to_copy(student_updates["decoder_layers"], teacher.config.decoder_layers)
        e_layers_to_copy: List = get_layers_to_copy(student_updates["encoder_layers"], teacher.config.encoder_layers)
        student_updates.update(self.teacher_kwargs)
        hparams.layer_to_copy = d_layers_to_copy
        hparams.e_layer_to_copy = e_layers_to_copy
        kw = teacher.config.to_diff_dict()
        kw.update(student_updates)
        # Copy weights
        student_cfg = BartConfig(**kw)
        student = BartForConditionalGeneration(student_cfg)
        student, _ = init_student(student, teacher)
        self.copy_to_student(d_layers_to_copy, e_layers_to_copy, hparams, student, teacher)
        Path(hparams.output_dir).mkdir(exist_ok=True)
        return d_layers_to_copy, student, student_cfg, teacher

    def copy_to_student(self, d_layers_to_copy, e_layers_to_copy, hparams, student, teacher):
        if teacher.config.model_type == "t5":
            return self.copy_t5_to_student(d_layers_to_copy, e_layers_to_copy, hparams, student, teacher)
        self.different_encoder: bool = hparams.student_encoder_layers != teacher.config.encoder_layers
        self.different_decoder = hparams.student_decoder_layers != teacher.config.decoder_layers
        if self.different_decoder:
            copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
        if self.different_encoder:
            copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)

    def copy_t5_to_student(self, d_layers_to_copy, e_layers_to_copy, hparams, student, teacher):
        self.different_encoder: bool = hparams.student_encoder_layers != teacher.config.num_layers
        self.different_decoder = hparams.student_decoder_layers != teacher.config.num_layers
        if self.different_decoder:
            copy_layers(teacher.decoder.block, student.decoder.block, d_layers_to_copy)
        if self.different_encoder:
            copy_layers(teacher.encoder.block, student.encoder.block, e_layers_to_copy)

    def get_dataset(self, type_path) -> SummarizationDataset:
        n_obs = self.n_obs[type_path]
        dataset = SummarizationDataset.from_raw_data(
            self.tokenizer, type_path=type_path, n_obs=n_obs, **self.dataset_kwargs
        )
        return dataset

    def _step(self, batch):
        # assert is_frozen(self.teacher)
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        # noinspection PyCallingNonCallable
        sloss, slogits, enc_outputs = self(
            source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,
        )
        loss_encoder = torch.tensor(0.0).type_as(sloss)
        loss_ce = torch.tensor(0.0).type_as(sloss)

        if self.different_encoder:
            with torch.no_grad():
                teacher_enc_outputs = self.teacher.model.encoder(source_ids, attention_mask=source_mask)
            if self.hparams.alpha_encoder_loss > 0:
                loss_encoder = self.calc_mse_loss(enc_outputs, teacher_enc_outputs[0], source_mask)
        else:
            teacher_enc_outputs = (enc_outputs,)
        with torch.no_grad():
            tloss, tlogits, *trash = self.teacher(
                source_ids,
                attention_mask=source_mask,
                encoder_outputs=teacher_enc_outputs,
                decoder_input_ids=y_ids,
                lm_labels=lm_labels,
            )
        dec_mask = invert_mask(self.model.model.last_padding_mask)
        if self.alpha_ce > 0:
            loss_ce, *_ = self.calc_ce_loss(dec_mask, slogits, tlogits)
        blended_loss = (
            loss_ce * self.alpha_ce + self.alpha_mlm * sloss + self.hparams.alpha_encoder_loss * loss_encoder
        )
        return blended_loss, loss_ce, sloss, loss_encoder

    def calc_mse_loss(self, teacher_outputs: torch.Tensor, student_outputs: torch.Tensor, mask) -> torch.FloatTensor:
        if mask is not None:
            # mask has False at padding_idx
            sel_mask = mask[:, :, None].expand_as(student_outputs).bool()
            s_logits_slct = torch.masked_select(student_outputs, sel_mask)
            t_logits_slct = torch.masked_select(teacher_outputs, sel_mask)
        else:
            t_logits_slct = teacher_outputs
            s_logits_slct = student_outputs
        return F.mse_loss(s_logits_slct, t_logits_slct)

    def calc_ce_loss(self, mask, s_logits, t_logits):
        if mask is not None:
            # mask has False at padding_idx
            sel_mask = mask[:, :, None].expand_as(s_logits)
            s_logits_slct = torch.masked_select(
                s_logits, sel_mask
            )  # (bs * seq_length * voc_size) modulo the 1s in mask
            t_logits_slct = torch.masked_select(
                t_logits, sel_mask
            )  # (bs * seq_length * voc_size) modulo the 1s in mask
        else:
            t_logits_slct = t_logits
            s_logits_slct = s_logits  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        return loss_ce, s_logits_slct, t_logits_slct

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        SummarizationTrainer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--teacher", default="facebook/bart-large-cnn", type=str,
        )
        parser.add_argument("--alpha_ce", default=0.8, type=float)
        parser.add_argument("--alpha_mlm", default=0.2, type=float)
        # parser.add_argument("--alpha_cos", default=0.0, type=float)
        parser.add_argument("--alpha_encoder_loss", default=0.0, type=float)
        parser.add_argument(
            "--alpha_hid", default=0.0, type=float, required=False,
        )
        parser.add_argument(
            "--student_decoder_layers", default=12, type=int, required=False,
        )
        parser.add_argument(
            "--student_encoder_layers", default=12, type=int, required=False,
        )
        parser.add_argument(
            "--no_teacher", action="store_true", default=False,
        )
        parser.add_argument(
            "--enc_only", action="store_true", default=False,
        )

        parser.add_argument("--auto_scale_batch_size", default=False, action="store_true")

        return parser


class EncoderDistiller(SummarizationDistiller):
    loss_names = ["loss"]

    def __init__(self, hparams):
        assert hparams.enc_only
        assert not hparams.no_teacher
        super().__init__(hparams)
        self.teacher_enc = self.teacher.model.encoder
        del self.teacher
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model.model.encoder(input_ids, attention_mask=attention_mask)

    def _step(self, batch):
        source_ids, source_mask, _ = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        enc_outputs = self(source_ids, source_mask)
        with torch.no_grad():
            teacher_enc_outputs = self.teacher_enc(source_ids, attention_mask=source_mask)
        loss_encoder = self.calc_mse_loss(enc_outputs[0], teacher_enc_outputs[0], source_mask)
        return (loss_encoder,)


class BrewerDistiller(SummarizationDistiller):
    loss_names = ["loss", "ce_loss", "mlm_loss", "enc_mse_loss", "hid_loss_enc", "hid_loss_dec"]
    teacher_kwargs = {"output_hidden_states": True}

    def _step(self, batch):
        # assert is_frozen(self.teacher)
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        decoder_input_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        # noinspection PyCallingNonCallable

        sloss, slogits, dec_hidden, enc_outputs, enc_hidden_state = self(
            source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

        def zero_tensor():
            return torch.tensor(0.0).type_as(sloss)

        loss_encoder, hid_loss_enc, hid_loss_dec = zero_tensor(), zero_tensor(), zero_tensor()
        if self.different_encoder:
            with torch.no_grad():
                teacher_enc_outputs, teacher_enc_hid, _ = self.teacher.model.encoder(
                    source_ids, attention_mask=source_mask
                )
            if self.hparams.alpha_encoder_loss > 0:
                loss_encoder = self.calc_mse_loss(enc_outputs, teacher_enc_outputs, source_mask)
            hid_loss_enc = self.calc_hidden_loss(
                source_mask, enc_hidden_state, teacher_enc_hid, self.hparams.e_layer_to_copy
            )
        else:
            teacher_enc_outputs = (enc_outputs,)

        with torch.no_grad():
            tloss, tlogits, tdec_hidden, _ = self.teacher(
                source_ids,
                attention_mask=source_mask,
                encoder_outputs=teacher_enc_outputs,
                decoder_input_ids=decoder_input_ids,
                lm_labels=lm_labels,
            )
        dec_mask = invert_mask(self.model.model.last_padding_mask)
        loss_ce, s_logits_slct, t_logits_slct = self.calc_ce_loss(dec_mask, slogits, tlogits)
        if not self.hparams.freeze_decoder and self.alpha_hid > 0:
            hid_loss_dec = self.calc_hidden_loss(dec_mask, dec_hidden, tdec_hidden, self.hparams.layer_to_copy)

        blended_loss = (
            loss_ce * self.alpha_ce
            + self.alpha_mlm * sloss
            + self.hparams.alpha_encoder_loss * loss_encoder
            + self.hparams.alpha_hid * (hid_loss_enc + hid_loss_dec)
        )
        return blended_loss, loss_ce, sloss, loss_encoder, hid_loss_enc, hid_loss_dec

    def calc_hidden_loss(self, attention_mask, hidden_states, hidden_states_T, matches):
        assert not isinstance(
            hidden_states, torch.Tensor
        ), f"expected list or tuple for hidden_states, got tensor of shape {hidden_states.shape}"
        assert not isinstance(
            hidden_states_T, torch.Tensor
        ), f"expected list or tuple for hidden_states_T, got tensor of shape {hidden_states_T.shape}"
        mask = attention_mask.to(hidden_states[0])
        valid_count = mask.sum() * hidden_states[0].size(-1)
        hidden_losses = [
            (F.mse_loss(hidden_states[i], hidden_states_T[j], reduction="none") * mask.unsqueeze(-1)).sum()
            / valid_count
            for i, j in enumerate(matches)
        ]
        return sum(hidden_losses)


class T5BrewerDistiller(BrewerDistiller):
    def pre_init(self, hparams):
        teacher = T5ForConditionalGeneration.from_pretrained(hparams.teacher, **self.teacher_kwargs).eval()
        n_layer = hparams.student_decoder_layers
        assert n_layer == hparams.student_encoder_layers  # TODO(SS): relax this
        d_layers_to_copy = get_layers_to_copy(n_layer, len(teacher.decoder.block))
        e_layers_to_copy: List = get_layers_to_copy(n_layer, len(teacher.encoder.block))
        student_updates = {"num_layers": n_layer}
        student_updates.update(self.teacher_kwargs)
        hparams.layer_to_copy = d_layers_to_copy
        hparams.e_layer_to_copy = e_layers_to_copy
        kw = teacher.config.to_diff_dict()
        kw.update(student_updates)
        # Copy weights
        student_cfg = T5Config(**kw)
        student = T5ForConditionalGeneration(student_cfg)
        student, _ = init_student(student, teacher)
        self.copy_to_student(d_layers_to_copy, e_layers_to_copy, hparams, student, teacher)
        Path(hparams.output_dir).mkdir(exist_ok=True)
        return d_layers_to_copy, student, student_cfg, teacher

    def freeze_embeds(self):
        freeze_part(self.model.shared)
        for d in [self.model.encoder, self.model.decoder]:
            # freeze_part(d.embed_positions)
            freeze_part(d.embed_tokens)

    def freeze_stuff(self, d_layers_to_copy):
        assert len(self.model.decoder.block) == len(d_layers_to_copy)
        assert_all_frozen(self.teacher)
        assert_all_frozen(self.model.decoder.embed_tokens)
        assert_all_frozen(self.model.encoder.embed_tokens)
        if self.different_encoder:
            assert any(grad_status(self.model.encoder))
        else:
            freeze_part(self.model.encoder)
            del self.teacher.model.encoder
        if self.different_decoder:
            assert any(grad_status(self.model.decoder))
        else:
            freeze_part(self.model.decoder)  # TODO(SS): very suspicious


def main(args):
    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    model: BaseTransformer = create_module(args)
    trainer: pl.Trainer = generic_train(model, args, early_stopping_callback=True)
    if not args.do_predict:
        return model
    # return model  # hack

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        # model = model.load_from_checkpoint(checkpoints[-1],
        #                                   #hparams_file=str(model.output_dir/'hparams.pkl')
        #                                   )
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)
    trainer.test(model)
    return model


def create_module(args) -> BaseTransformer:
    t5 = "t5" in args.model_name_or_path
    if args.no_teacher:
        assert not args.enc_only
        module_cls = SummarizationTrainer
    elif t5:
        module_cls = T5BrewerDistiller
    elif args.enc_only:
        module_cls = EncoderDistiller
    elif args.alpha_hid > 0:
        module_cls = BrewerDistiller
    else:
        module_cls = SummarizationDistiller
    args.setup_cls: str = module_cls.__name__
    model = module_cls(args)
    return model


def eval_and_fix(args):
    Path(args.output_dir).mkdir(exist_ok=True)
    model: BaseTransformer = create_module(args)
    trainer: pl.Trainer = generic_train(model, args, early_stopping_callback=False)
    trainer.test(model)


def evaluate_checkpoint(ckpt_path: Path, dest_dir=None):
    exp_dir = ckpt_path.parent
    if dest_dir is None:
        dest_dir = exp_dir
    clash = list(dest_dir.glob("test_generations*"))
    if clash:
        print(f"SKIPPING to avoid overwriting {clash}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "hparams" in ckpt:
        args = argparse.Namespace(**ckpt["hparams"])
    else:
        args = argparse.Namespace(**pickle_load(exp_dir / "hparams.pkl"))
    args.resume_from_checkpoint = str(ckpt_path)
    args.do_train = False
    args.output_dir = str(dest_dir)
    args.n_gpu = 1
    args.eval_batch_size = 16
    return eval_and_fix(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = SummarizationDistiller.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    main(args)
