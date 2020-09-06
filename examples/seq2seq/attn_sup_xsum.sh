#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export WANDB_PROJECT=transformers_fork-examples_summarization_bart
python -m pdb -c continue distillation.py \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 \
  --val_check_interval 0.2 --n_val 1000 --eval_beams 2 \
  --teacher facebook/bart-large-xsum \
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --student_decoder_layers 3 --student_encoder_layers 12 \
  --freeze_encoder --freeze_embeds \
  --model_name_or_path IGNORED \
  --length_penalty=0.5 \
  --train_batch_size=$BS --eval_batch_size=$BS --gradient_accumulation_steps=$GAS  --num_train_epochs=4 \
  --tokenizer_name facebook/bart-large \
  --warmup_steps 500 --logger_name wandb --sortish_sampler --gpus 1 \
  "$@"
