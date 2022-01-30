import argparse
from time import time
import os

import pandas as pd
from scipy import stats
import numpy as np

def evaluate_results(df, dict_results, significance=0.01):
    df = df.dropna(axis=1, how='all')
    #list=['map','prec@1','prec@5','prec@10', 'recall@1', 'recall@5', 'recall@10']
    test_df = pd.DataFrame(columns= ['metric','baseline','estimate','pvalue','reject'])
    print('\t One Sample t-test')
    print("alternative hypothesis: estimate mean is not equal to baseline mean.")
    print("results:")
    for column in df.columns:
        pvalue = round(stats.ttest_1samp(df[column], dict_results[column]).pvalue, 4)
        reject = True if pvalue<significance else False
        est_col_mean=round(np.mean(df[column]), 4)
        output_str = "%s: pvalue=%.4f, estimated_mean=%.4f, baseline=%.4f, reject_null(@%.2f)=%s" % (column,pvalue,est_col_mean,dict_results[column],significance,reject)
        print(output_str)
        test_df = test_df.append(pd.DataFrame([[column,dict_results[column],est_col_mean,pvalue,reject]],columns=['metric','baseline','estimate','pvalue','reject']), ignore_index=True)
    return test_df

def evaluate_datasets(save_dir,baseline):
    directory = os.fsencode(save_dir)
    for d in os.listdir(directory):
        eval_dir = os.fsdecode(d)
        results_dir = save_dir + eval_dir + '/results.csv'
        print(results_dir)
        column_names = ['map','prec@1','prec@5','prec@10', 'recall@1', 'recall@5', 'recall@10']
        if not os.path.exists(results_dir):
            continue
        results = pd.read_csv(results_dir) 
        print('*** %s ***' % (eval_dir))
        evaluate_results(results, baseline[eval_dir])
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='checkpoints/',
                        help='path to save checkpoints')
    args = parser.parse_args()
    baseline = {'ml1m':{'map':0.1883, 'prec@1':0.3308, 'prec@5':0.2831, 'prec@10':0.2493, 'recall@1':0.0202, 'recall@5':0.0843, 'recall@10':0.1438},
                'gowalla':{'map':0.098, 'prec@1':0.2135, 'prec@5':0.1190, 'prec@10':0.0884, 'recall@1':0.0337, 'recall@5':0.0890, 'recall@10':0.1305},
                'ml1m-custom':{'map':0.1883, 'prec@1':0.3308, 'prec@5':0.2831, 'prec@10':0.2493, 'recall@1':0.0202, 'recall@5':0.0843, 'recall@10':0.1438},
                'gowalla-custom':{'map':0.098, 'prec@1':0.2135, 'prec@5':0.1190, 'prec@10':0.0884, 'recall@1':0.0337, 'recall@5':0.0890, 'recall@10':0.1305}}
    evaluate_datasets(args.save_root,baseline)
