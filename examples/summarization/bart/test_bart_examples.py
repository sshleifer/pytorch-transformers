import argparse
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader

from durbango import DEFAULT_DEVICE, lmap, pickle_load, pickle_save
from transformers import BartTokenizer

from .evaluate_cnn import generate_summaries, run_generate
from .finetune import eval_and_fix, evaluate_checkpoint, main
from .utils import PSEUDO_ID_SUFFIX, SummarizationDataset, clean_output_dir, summaries_for_file


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()

CHEAP_ARGS = {
    "alpha_hid": 0,
    "enc_only": False,
    "tgt_suffix": "",
    "resume_from_checkpoint": None,
    "sortish_sampler": True,
    "student_decoder_layers": 1,
    "val_check_interval": 1.0,
    "output_dir": "",
    "fp16": False,
    "no_teacher": False,
    "fp16_opt_level": "O1",
    "gpus": 0,
    "n_tpu_cores": 0,
    "max_grad_norm": 1.0,
    "do_train": True,
    "do_predict": True,
    "gradient_accumulation_steps": 1,
    "server_ip": "",
    "server_port": "",
    "seed": 42,
    "model_type": "bart",
    "model_name_or_path": "sshleifer/bart-tiny-random",
    "config_name": "",
    "tokenizer_name": "facebook/bart-large",
    "cache_dir": "",
    "do_lower_case": False,
    "learning_rate": 3e-05,
    "weight_decay": 0.0,
    "adam_epsilon": 1e-08,
    "warmup_steps": 0,
    "num_train_epochs": 1,
    "train_batch_size": 2,
    "eval_batch_size": 2,
    "max_source_length": 12,
    "max_target_length": 12,
    "val_mtl": 12,
    "test_mtl": 12,
    "fast_dev_run": False,
    "no_cache": False,
    "n_train": -1,
    "n_val": -1,
    "n_test": -1,
    "student_encoder_layers": 1,
    "alpha_loss_encoder": 0.0,
    "freeze_encoder": False,
    "freeze_decoder": False,
    "auto_scale_batch_size": False,
}


def _dump_articles(path: Path, articles: list):
    with path.open("w") as f:
        f.write("\n".join(articles))


BDIR = Path("~/transformers_fork/examples/summarization/bart/").absolute()

pseudo_targets_path = BDIR / "pseudo_target_ids.pkl"
PSEUDO_IDS = [
    [0, 39762, 12, 23822, 1329, 11777, 4831, 305, 5867, 16],
    [0, 37038, 815, 10433, 16, 6146, 63, 291, 212, 191],
]


def make_test_data_dir():
    tmp_dir = Path(tempfile.gettempdir())
    articles = [" Sam ate lunch today", "Sams lunch ingredients"]
    summaries = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]
    for split in ["train", "val", "test"]:
        _dump_articles((tmp_dir / f"{split}.source"), articles)
        _dump_articles((tmp_dir / f"{split}.target"), summaries)
        _dump_articles((tmp_dir / f"{split}.target.pseudo"), summaries)
        pickle_save(PSEUDO_IDS, (tmp_dir / f"{split}_{PSEUDO_ID_SUFFIX}"))
    return tmp_dir


class TestBartExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        logging.disable(logging.CRITICAL)  # remove noisy download output from tracebacks
        return cls

    def test_bart_cnn_cli(self):
        tmp = Path(tempfile.gettempdir()) / "utest_generations_bart_sum.hypo"
        output_file_name = Path(tempfile.gettempdir()) / "utest_output_bart_sum.hypo"
        articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]
        _dump_articles(tmp, articles)
        testargs = ["evaluate_cnn.py", str(tmp), str(output_file_name), "sshleifer/bart-tiny-random"]
        with patch.object(sys, "argv", testargs):
            run_generate()
            self.assertTrue(Path(output_file_name).exists())
            os.remove(Path(output_file_name))

    @unittest.skipUnless(torch.cuda.device_count() > 1, "skipping multiGPU test")
    def test_bdc_multigpu(self):
        updates = dict(
            student_encoder_layers=2,
            student_decoder_layers=1,
            no_teacher=True,
            freeze_encoder=True,
            gpus=2,
            sortish_sampler=False,
        )
        self._bart_distiller_cli(updates)

    def test_bdc_no_teacher(self):
        updates = dict(
            student_encoder_layers=2,
            student_decoder_layers=1,
            no_teacher=True,
            tgt_suffix=".pseudo",
            freeze_decoder=True,
        )
        self._bart_distiller_cli(updates)

    def test_bdc_yes_teacher(self):
        updates = dict(student_encoder_layers=2, student_decoder_layers=1,)
        self._bart_distiller_cli(updates)

    def test_bdc_checkpointing(self):

        updates = dict(
            student_encoder_layers=2, student_decoder_layers=1, num_train_epochs=4, val_check_interval=0.25,
        )
        model = self._bart_distiller_cli(updates, check_contents=False)

        ckpts = list(Path(model.output_dir).glob("*.ckpt"))
        self.assertEqual(1, len(ckpts))
        transformer_ckpts = list(Path(model.output_dir).glob("**/*.bin"))
        self.assertEqual(len(transformer_ckpts), len(ckpts))
        n_extra = 0  # more than 1 checkpoint saved, specified in lightning_base.py
        # removed = clean_output_dir(model.output_dir)
        # elf.assertEqual(len(removed), n_extra)
        new_transformer_ckpts = list(Path(model.output_dir).glob("**/*.bin"))
        self.assertEqual(len(new_transformer_ckpts), 1)
        examples = lmap(str.strip, model.hparams.data_dir.joinpath("test.source").open().readlines())
        out_path = tempfile.mktemp()
        generate_summaries(examples, out_path, model_name=new_transformer_ckpts[0].parent)
        self.assertTrue(Path(out_path).exists())

        evaluate_checkpoint(ckpts[0], dest_dir=Path(tempfile.mkdtemp()))

    def test_bdc_brewer(self):
        updates = dict(student_encoder_layers=1, student_decoder_layers=1, alpha_hid=2.0)
        self._bart_distiller_cli(updates)

    def test_bdc_t5(self):
        updates = dict(
            student_encoder_layers=1,
            student_decoder_layers=1,
            alpha_hid=2.0,
            teacher="patrickvonplaten/t5-tiny-random",
            model_type="t5",
            model_name_or_path="patrickvonplaten/t5-tiny-random",
            tokenizer_name="t5-small",
        )
        self._bart_distiller_cli(updates)

    @unittest.skipUnless(False, "Not implemented")
    def test_bdc_mbart(self):
        pass

    def test_bdc_enc_only(self):
        updates = dict(alpha_mlm=0.0, alpha_ce=0.0, student_encoder_layers=1, enc_only=True, student_decoder_layers=2,)
        model = self._bart_distiller_cli(updates)
        self.assertFalse(model.different_decoder)
        ckpt_path = list(model.output_dir.glob("*.ckpt"))[0]
        evaluate_checkpoint(ckpt_path, dest_dir=Path(tempfile.mkdtemp(prefix="output_v2")))

    def _bart_distiller_cli(self, updates, check_contents=True):
        default_updates = dict(
            model_type="bart",
            train_batch_size=1,
            eval_batch_size=2,
            num_train_epochs=2,
            alpha_mlm=0.2,
            alpha_ce=0.8,
            do_predict=True,
            gpus=0,
            model_name_or_path="sshleifer/tinier_bart",
            teacher=CHEAP_ARGS["model_name_or_path"],
            val_check_interval=0.5,
            alpha_encoder_loss=0.4,
        )
        default_updates.update(updates)
        args_d: dict = CHEAP_ARGS.copy()
        tmp_dir = make_test_data_dir()
        output_dir = tempfile.mkdtemp(prefix="output_")

        args_d.update(data_dir=tmp_dir, output_dir=output_dir, **default_updates)
        model = main(argparse.Namespace(**args_d))
        if not check_contents:
            return model
        contents = os.listdir(output_dir)
        ckpt_name = "val_avg_rouge2=0.0000-step_count=2.ckpt"  # "val_avg_rouge2=0.0000-epoch=1.ckpt"  # "epoch=1-val_avg_rouge2=0.0000.ckpt"
        contents = {os.path.basename(p) for p in contents}
        self.assertIn(ckpt_name, contents)
        self.assertIn("metrics.pkl", contents)
        self.assertIn("test_generations.txt", contents)
        self.assertIn("val_generations_3.txt", contents)
        self.assertIn("val_3_results.txt", contents)
        self.assertIn("test_results.txt", contents)
        self.assertEqual(len(contents), 15)

        metrics = pickle_load(Path(output_dir) / "metrics.pkl")
        val_df = pd.DataFrame(metrics["val"])
        train_df = pd.DataFrame(metrics["train"])
        test_df = pd.DataFrame(metrics["test"])
        desired_n_evals = args_d["num_train_epochs"] * 2 + 1
        self.assertEqual(val_df.shape[0], desired_n_evals)  #
        self.assertEqual(test_df.shape[1], val_df.shape[1])
        self.assertEqual(train_df.shape[0], 0)
        return model

    def test_t5_run_sum_cli(self):
        args_d: dict = CHEAP_ARGS.copy()
        tmp_dir = make_test_data_dir()
        output_dir = tempfile.mkdtemp(prefix="output_")
        args_d.update(
            data_dir=tmp_dir,
            model_type="t5",
            model_name_or_path="patrickvonplaten/t5-tiny-random",
            train_batch_size=2,
            eval_batch_size=2,
            gpus=0,
            output_dir=output_dir,
            do_predict=True,
        )
        main(argparse.Namespace(**args_d))

    def test_bart_summarization_dataset(self):
        tmp_dir = Path(tempfile.gettempdir())
        articles = [" Sam ate lunch today", "Sams lunch ingredients"]
        summaries = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]
        _dump_articles((tmp_dir / "train.source"), articles)
        _dump_articles((tmp_dir / "train.target"), summaries)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        max_len_source = max(len(tokenizer.encode(a)) for a in articles)
        max_len_target = max(len(tokenizer.encode(a)) for a in summaries)
        trunc_target = 4
        train_dataset = SummarizationDataset(
            tokenizer, data_dir=tmp_dir, type_path="train", max_source_length=20, max_target_length=trunc_target,
        )
        dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
        for batch in dataloader:
            self.assertEqual(batch["source_mask"].shape, batch["source_ids"].shape)
            # show that articles were trimmed.
            self.assertEqual(batch["source_ids"].shape[1], max_len_source)
            self.assertGreater(20, batch["source_ids"].shape[1])  # trimmed significantly

            # show that targets were truncated
            self.assertEqual(batch["target_ids"].shape[1], trunc_target)  # Truncated
            self.assertGreater(max_len_target, trunc_target)  # Truncated

    def test_summaries_for_file(self):

        tmp_dir = Path(tempfile.gettempdir())

        self.needs_remove = tmp_dir
        articles = [" Sam ate lunch today", "Sams lunch ingredients"]
        summaries = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]
        _dump_articles((tmp_dir / "train.source"), articles)
        _dump_articles((tmp_dir / "train.target"), summaries)

        summary_ids, summary_text = summaries_for_file(
            "sshleifer/bart-tiny-random", "train", data_dir=tmp_dir, bs=1, max_source_length=10, max_target_length=4,
        )

        self.assertEqual(len(summary_ids), len(articles))
        # self.assertEqual(summary_ids.shape, len(articles))

    def tearDown(self) -> None:
        import shutil

        if not hasattr(self, "needs_remove"):
            return
        data_dir = self.needs_remove
        type_path = "train"
        to_rm = [data_dir / f"{type_path}_pseudo_ids.pkl", data_dir / f"{type_path}_pseudo_ids.pkl"]
        for p in to_rm:
            if p.exists():
                os.remove(p)


def list_to_text_file(lst, path):
    dest = Path(path)
    dest.open("w+").writelines(lst)
