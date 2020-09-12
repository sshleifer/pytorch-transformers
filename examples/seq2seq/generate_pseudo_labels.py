import argparse
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
import fire

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader


logger = getLogger(__name__)

try:
    from .utils import calculate_bleu, calculate_rouge, parse_numeric_cl_kwargs, use_task_specific_params, Seq2SeqDataset, save_json
    from .process_pseudolabels import process_pseudolabels
except ImportError:
    from utils import calculate_bleu, calculate_rouge, parse_numeric_cl_kwargs, use_task_specific_params, Seq2SeqDataset, save_json
    from process_pseudolabels import process_pseudolabels

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_pseudolabels(
    data_dir,
    out_file: str,
    model_name: str,
    bs: int = 8,
    max_source_length: int=1024,
    device: str = DEFAULT_DEVICE,
    n_obs=None,
    fp16=False,
    num_return_sequences:int=10,
    num_beams: int=10,
    task="summarization",
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    #fout = Path(out_file).open("w", encoding="utf-8")
    Path(out_file).parent.mkdir(exist_ok=True)
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    use_task_specific_params(model, task)
    ds = Seq2SeqDataset(
        tokenizer,
        data_dir,
        max_source_length, max_target_length=1024,
        type_path='train',
        n_obs=n_obs,
        prefix=model.config.prefix,
    )
    sampler = ds.make_sortish_sampler(bs)

    data_loader = DataLoader(
        ds,
        sampler=sampler,
        batch_size=bs,
        collate_fn=ds.collate_fn

    )
    # update config with task specific params
    i = 0
    results = []
    for batch in tqdm(data_loader):
        i+=1

        summaries = model.generate(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            **generate_kwargs,
        )
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        chunked_preds = list(chunks(dec, num_return_sequences))
        for i, label in enumerate(labels):
            results.append(dict(preds=chunked_preds[i], label=label))
        save_json(results, out_file)

    process_pseudolabels(out_file)


if __name__ == '__main__':
    fire.Fire(generate_pseudolabels)
