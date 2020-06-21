
from durbango import *
from torch.utils.data import DataLoader
ds = None
tok = None
def runner(model, max_i=500, bs=8, ds=ds, **gen_kwargs):
    dataloader = DataLoader(
    ds,
    batch_size=bs,
    collate_fn=ds.collate_fn,
    shuffle=False,
    num_workers=4
    )
    summaries = []
    scores = []
    for i, batch in tqdm_nice(enumerate(dataloader)):
        yids = batch.pop('decoder_input_ids')
        batch = {k: v.to(DEFAULT_DEVICE) for k,v in batch.items()}
        sents, bscores = model.generate(**batch, **gen_kwargs)
        dec = tok.batch_decode(sents, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        summaries.extend(dec)
        scores.extend(bscores)
        if i >= max_i:
            break
    return scores, summaries

def make_stat_df(model, dmodel, **gen_kwargs):
    sco1, summ1 = runner(model, **gen_kwargs)
    sco2, summ2 = runner(dmodel, **gen_kwargs)
    lens1 = lmap(len, summ1)
    lens2 = lmap(len, summ2)
    test_stat_df = pd.DataFrame(dict(l1=lens1, s1=sco1, s2=sco2, l2=lens2, summ1=summ1, summ2=summ2))
    rbig = calculate_rouge_score(summ1, tgt_lns)
    rsmall = calculate_rouge_score(summ2, tgt_lns)
    print(f'Rouge big: {rbig}')
    print(f'Rouge small: {rsmall}')
    return test_stat_df
make_stat_df(model, dmodel, num_beams=1)
