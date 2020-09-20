### Pseudolabels

Generate pseudolabels with the `google/pegasus-xsum` model on the cnn data.
I do not believe that `--max_source_length` nor `--length_penalty` are needed

```bash
python -m torch.distributed.launch --nproc_per_node=2  run_distributed_eval.py \
	--model_name google/pegasus-xsum --data_dir cnn_dm --max_source_length 512 --type_path train \
	--save_dir pegasus_pls_from_xsum_on_cnn --bs 16 --task summarization --length_penalty 0.6 --debug \
	--sync_timeout 12000
```
