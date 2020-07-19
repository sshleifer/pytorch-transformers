#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
python distillation.py \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 \
  --val_check_interval 0.25 \
  --teacher facebook/mbart-large-en-ro \
  --freeze_encoder --freeze_embeds --data_dir $ENRO_DIR \
  --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
  --freeze_encoder --freeze_embeds \
  --train_batch_size=$BS --eval_batch_size=$BS --gradient_accumulation_steps=$GAS --num_train_epochs=3 \
  --tokenizer_name facebook/mbart-large-cc25 --src_lang en_XX --tgt_lang ro_RO \
  --task translation \
  --warmup_steps 500 \
  $@
