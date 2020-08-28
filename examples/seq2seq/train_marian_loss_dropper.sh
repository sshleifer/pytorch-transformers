#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export WANDB_PROJECT=dmar
# export MAX_LEN=128
m=Helsinki-NLP/opus-mt-en-ro
export MAX_LEN=128
python finetune.py \
  --learning_rate=3e-4 \
  --loss_dropper 0.3 \
  --do_train \
  --fp16 \
  --val_check_interval 0.25 \
  --data_dir wmt_en_ro \
  --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
  --train_batch_size=$BS --eval_batch_size=$BS \
  --tokenizer_name $m --model_name_or_path $m \
  --warmup_steps 500 --sortish_sampler --logger_name wandb \
  --gpus 1 --fp16_opt_level=O1 --task translation \
  "$@"
