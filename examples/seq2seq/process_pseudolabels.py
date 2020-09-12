import argparse
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader


logger = getLogger(__name__)
from rouge_score import rouge_scorer
try:
    from .utils import load_json, pickle_save
except ImportError:
    from utils import  load_json, pickle_save



import numpy as np

import fire

def process_pseudolabels(path, save_dir=None, n_obs=None):
    if save_dir is None:
        save_dir = Path(path).parent
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    results = load_json(path)
    if n_obs is not None:
        results = results[:n_obs]
    f = save_dir.joinpath('train.target').open("w", encoding="utf-8")
    METRIC='rougeL'
    scorer = rouge_scorer.RougeScorer([METRIC], use_stemmer=False)
    rouges = []
    best_preds = []
    labs = []
    for r in tqdm(results):
        lab, preds = r['label'], r['preds']
        possible_rouges = list(enumerate([scorer.score(p, lab)[METRIC].fmeasure for p in preds]))
        best_id, best_rouge = sorted(possible_rouges, key=lambda x: -x[1])[0]
        rouges.append(best_rouge)
        best_preds.append(preds[best_id])
        labs.append(lab)
        f.write(preds[best_id]+'\n')
        f.flush()
    f.close()
    print(f'RougeL: {np.mean(rouges): .2f}')
    pickle_save(rouges, save_dir.joinpath('train_rouges.pkl'))



if __name__ == '__main__':
    fire.Fire(process_pseudolabels)
