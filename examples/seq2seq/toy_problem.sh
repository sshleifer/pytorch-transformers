#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export WANDB_PROJECT=dmar
# export MAX_LEN=128
python distillation.py \
  --learning_rate=6e-4 \
  --do_train \
  --fp16 \
  --val_check_interval 0.25 \
  --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
  --student_decoder_layers 1 --student_encoder_layers 6 \
  --freeze_encoder --freeze_embeds \
  --model_name_or_path IGNORED \
  --alpha_hid=3. \
  --train_batch_size=32 --eval_batch_size=32 \
  --tokenizer_name Helsinki-NLP/opus-mt-en-ro \
  --warmup_steps 500 --logger_name wandb \
  --fp16_opt_level O1 --task translation --normalize_hidden --num_sanity_val_steps=0 --supervise_forward \
  "$@"
