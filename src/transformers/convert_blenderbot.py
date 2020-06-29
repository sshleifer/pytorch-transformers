# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Blenderbot checkpoint."""

import argparse
import logging
import os
from pathlib import Path

import torch

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
    BlenderbotConditionalGeneration,
    BlenderbotConfig,
)
from transformers.modeling_bart import _make_linear_from_emb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

replacers = [
    # ['self_attention', 'self_attn'],
    ["attention", "attn"],
    ["encoder_attention", "encoder_attn"],
    ["q_lin", "q_proj"],
    ["k_lin", "k_proj"],
    ["v_lin", "v_proj"],
    ["out_lin", "out_proj"],
    ["norm_embeddings", "layernorm_embedding"],
    ["position_embeddings", "embed_positions"],
    ["embeddings", "embed_tokens"],
    ["ffn.lin", "fc"],
]
hard_code = {"embeddings.weight": "shared.weight"}


def remap(k):
    if k in hard_code:
        return hard_code[k]

    for a, b in replacers:
        k = k.replace(a, b)

    if k.startswith("encoder"):
        k = k.replace(".attn", ".self_attn")
        k = k.replace("norm1", "self_attn_layer_norm")
        k = k.replace("norm2", "final_layer_norm")
    elif k.startswith("decoder"):
        k = k.replace("norm1", "self_attn_layer_norm")
        k = k.replace("norm2", "encoder_attn_layer_norm")
        k = k.replace("norm3", "final_layer_norm")
    return k


IGNORE_KEYS = ["START"]


@torch.no_grad()
def convert_parlai_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_json_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    model = torch.load("blenderbot_model.bin", map_location="cpu")
    sd = model["model"]
    cfg = BlenderbotConfig.from_json_file("blenderbot-3B-config.json")
    m = BlenderbotConditionalGeneration(cfg)
    failures = []
    mapping = {}
    for k, v in sd.items():
        if k in IGNORE_KEYS:
            continue
        new_k = remap(k)
        if new_k not in m.state_dict():
            failures.append([k, new_k])
        else:
            mapping[new_k] = v
    m.load_state_dict(mapping, strict=True)
    m.half()
    m.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("parlai_path", type=str, help="like blenderbot-model.bin")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config_json", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum"
    )
    args = parser.parse_args()
    convert_parlai_checkpoint(args.parlai_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config_json)
