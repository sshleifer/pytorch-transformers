import pandas as pd

from pathlib import Path
import numpy as np
PEG_CNN_EXP = {
    'cnn_no_t_142_v2': 'pegasus-16-16',
    'dpx_cnn_4_no_t_cont_142': 'dPegasus-16-4',
    'dl6_no_teacher': 'dBart-12-6',
    'bart-large-cnn': 'BART-12-12',
    'cnn_no_teacher_f12_3': 'dBART-12-3',
    'bertabs-6-6': 'bertabs-6-6',
}

record = pd.DataFrame(
    {
        'bart-large-cnn': {'R2': 21.06, 'RL': 30.63},
    'bertabs-6-6': {'R2': 19.58, 'RL': np.nan}
     }
).T

recordx = pd.DataFrame(
    {
    'bertabs-6-6': {'R2': 16.58, 'RL': np.nan}}
).T
STALE_RESULTS = ['xsum_baseline']
from utils import load_json
def test_rouge_df():
    files = Path('.').glob('**/test_rouge.json')
    records = []
    for p in files:
        dct = load_json(p)
        dct['fname'] = p.parent.name
        avg = (dct['fname'] == 'avg_tfmr_metrics')
        if avg:
            dct['fname'] = p.parent.parent.name + '_avg'
        dct['avg'] = avg
        records.append(dct)
    #pd.Series(
    df = pd.DataFrame(records).set_index('fname')
    return df
def is_definitely_pegasus(fname):
    if 'dpx' in fname or '1212' in fname:
        return True
    return False

def make_clean_bart_xsum_table(df, max_r2_score=22.7):
    xsum_df = df[df.n_obs==11333.].drop('n_gpus', 1)[['rouge2', 'rougeL', 'avg']].dsort()
    xsum_df['dpx'] = xsum_df.index.map(is_definitely_pegasus)

    # Query bart experiments with no checkpoint averaging
    bart_xdf = xsum_df.loc[(~xsum_df.dpx) & (xsum_df.rouge2 < max_r2_score)].drop('dpx', 1).loc[lambda x: ~x.avg].drop('avg', 1)

    legacy_results = pd.read_csv('rouge_test_xsum.csv').set_index('exp_name')
    combined = pd.concat([
        bart_xdf.rename(columns=lambda x: x.replace('rouge', 'R')),
        legacy_results[['R2', 'RL']] * 100
    ]
    ).dsort().round(4).drop([STALE_RESULTS], errors='ignore')
    return combined

def make_clean_bart_cnn_table(df, max_r2_score=22.7):
    xsum_df = df[df.n_obs==11490].drop('n_gpus', 1)[['rouge2', 'rougeL', 'avg']].dsort()
    xsum_df['dpx'] = xsum_df.index.map(is_definitely_pegasus)

    # Query bart experiments with no checkpoint averaging
    bart_xdf = xsum_df.loc[(xsum_df.rouge2 < max_r2_score)].drop('dpx', 1).loc[lambda x: ~x.avg].drop('avg', 1)

    legacy_results = pd.read_csv('rouge_test_cnn.csv').set_index('exp_name')
    combined = pd.concat([
        bart_xdf.rename(columns=lambda x: x.replace('rouge', 'R')),
        legacy_results[['R2', 'RL']] * 100
    ]
    ).dsort().round(4).drop([STALE_RESULTS], errors='ignore')
    return combined
df = test_rouge_df()
cnn_tab = make_clean_bart_cnn_table(df)
cnn_tab_bart = cnn_tab.append(record).dsort()

PEG_XSUM_EXP = {
    'gpx_baseline': 'Pegasus',
    '1212_fp32_cc2': 'P-12-12',
    'dpx_4_from_large': 'P-16-4',
    'xsum_bart_baseline_sep20': 'BART',
    'dbart_xsum_12_6_metrics': 'BART-12-6',
}

PEG_CNN_EXP = {
    'cnn_no_t_142_v2': 'pegasus',
    'dpx_cnn_4_no_t_cont_142': 'P-16-4',
    'dl6_no_teacher': 'B-12-6',
    'bart-large-cnn': 'BART',
    'cnn_no_teacher_f12_3': 'B-12-3',
    'bertabs-6-6': "BERTABS-6-6"
}

cnn_part = cnn_tab_bart.loc[list(PEG_CNN_EXP)].assign(data='CNN/DM')#.reset_index()
combined = make_clean_bart_xsum_table(df)
bart_xsum = combined.loc[combined.index.isin(PEG_XSUM_EXP)].append(recordx).assign(data='XSUM')
pegasus_xsum = df.loc[list(PEG_XSUM_EXP.keys())].rename(columns={'rouge2': 'R2'}).rename(index=PEG_XSUM_EXP)[['R2']].assign(data='XSUM')
pl1 = pd.concat([pegasus_xsum, cnn_part]).drop('RL', 1).rename_axis('model')

# One observation of length 512 from xsum dataset
TIME_DATA = {'google/pegasus-cnn_dailymail': 1.4916868209838867,
             'sshleifer/pegasus-cnn-ft-v2': 1.3740575313568115,
             'facebook/bart-large-cnn': 0.9272456169128418,
             'sshleifer/distill-pegasus-cnn-16-4': 0.8749051094055176,
             'google/pegasus-xsum': 0.788020133972168,
             'sshleifer/distill-pegasus-xsum-16-8': 0.69443678855896,
             'sshleifer/distilbart-cnn-12-6': 0.6206557750701904,
             'sshleifer/distilbart-cnn-6-6': 0.6183633804321289,
             'sshleifer/distill-pegasus-xsum-16-4': 0.5872972011566162,
             'facebook/bart-large-xsum': 0.5198006629943848,
             'sshleifer/distilbart-cnn-12-3': 0.4680106639862061,
             'sshleifer/distilbart-xsum-9-6': 0.3991215229034424,
             'sshleifer/distilbart-xsum-6-6': 0.3966751098632813,
             'sshleifer/distilbart-xsum-12-6': 0.3917369842529297,
             'sshleifer/distilbart-xsum-12-3': 0.3051614761352539,
             'sshleifer/distilbart-xsum-12-1': 0.2734696865081787,
             'sshleifer/distilbart-xsum-1-1': 0.2646820545196533,
             'bertabs-6-6-cnn': .58,
             'bertabs-6-6-xsum': .390
             }

wandb_to_hub_renamer = {
    'gpx_baseline': 'google/pegasus-xsum',
    '1212_fp32_cc2': 'sshleifer/distill-pegasus-xsum-16-4',
    'dpx_4_from_large': 'sshleifer/distill-pegasus-xsum-16-4',
    'xsum_bart_baseline_sep20': "facebook/bart-large-xsum",
    'dbart_xsum_12_6_metrics': 'sshleifer/distilbart-xsum-12-6',
    'cnn_no_t_142_v2': 'sshleifer/pegasus-cnn-ft-v2',
    'dpx_cnn_4_no_t_cont_142': 'sshleifer/distill-pegasus-cnn-16-4',
    'dl6_no_teacher': 'sshleifer/distilbart-xsum-12-3',
    'bart-large-cnn': 'facebook/bart-large-cnn',
    'cnn_no_teacher_f12_3': 'sshleifer/distilbart-cnn-12-3',
    'bert-abs-cnn-6-6': 'bertabs-cnn-6-6',
    'bert-abs-xsum-6-6': 'bertabs-xsum-6-6',
}
def plot_name(wandb_name):
    return wandb_name.str.lower().replace('bart', 'B').replace('pegasus', 'p')

def get_inference_time(wandb_name):
    hub_name = wandb_to_hub_renamer[wandb_name]
    return TIME_DATA[hub_name]
