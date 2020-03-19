import unittest

import torch

from tests.utils import require_torch, slow
from transformers import BartTokenizer, BartModel

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@require_torch
class MemoryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        source_path = "test.source"
        cls.lns = [" " + x.rstrip() for x in open(source_path).readlines()][:6]
        tokenizer = BartTokenizer.from_pretrained('bart-large')
        dct = tokenizer.batch_encode_plus(cls.lns, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        cls.ids = dct['input_ids'].to(DEFAULT_DEVICE)

    def test_base_model_mem(self):
        model = BartModel.from_pretrained('bart-large').to(DEFAULT_DEVICE)
        model.log_mem('after init', verbose=True)
        model.reset_logs()
        model(self.ids)
        log_df = model.combine_logs()
        log_df.to_csv('hf_batch_fwd_logs.csv')
        print(model.summary)
