import argparse
from time import time
import os

import pandas as pd
from scipy import stats
import numpy as np

def preprocess(df, dataset, threshold, train_size=0.7):
    if dataset=='gowalla':
        df.columns =['user_id', 'timestamp', 'latitude', 'longitude', 'location'] 	
        df = df.drop(columns=['latitude'])
        df['longitude'] = 1
        df = df[["user_id", "location", "longitude"]]
    elif dataset=='ml1m':
        df.columns =['user_id', 'item_id', 'rating', 'timestamp']
        df['rating'] = 1
    else:
        raise Exception("Datasets available: gowalla, ml1m")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df =df.sort_values(by=['timestamp'])
    df = df.drop(columns=['timestamp'])

    df = df.reset_index(drop=True)
    df["user_id"] = pd.to_numeric(df["user_id"])

    counts_col = df.groupby("user_id")["user_id"].transform(len)
    mask = (counts_col > threshold)
    df =df[mask]

    train = pd.DataFrame()
    test = pd.DataFrame()

    for i in range(df['user_id'].min(),df['user_id'].max()+1,1):
        df4= df[df['user_id']== i]
        nu = len(df[df['user_id']== i])*train_size
        number_train= round(nu)
        number_test = len(df[df['user_id']== i])- round(nu)
        train = train.append(df4.head(number_train), ignore_index = True)
        test = test.append(df4.tail(number_test), ignore_index = True)

    save_dir = f'data/{dataset}-custom/test/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('created di')
    train.to_csv(save_dir+'train.txt', header=None, index=None, sep=' ', mode='a')
    test.to_csv(save_dir+'test.txt', header=None, index=None, sep=' ', mode='a')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='checkpoints/',
                        help='path to save checkpoints')
    parser.add_argument('--dataset', type=str, required=True,
                            choices=['ml1m', 'gowalla'])
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()
    if args.dataset=='ml1m':
        df = pd.read_csv("raw_data/ratings.dat", sep="::",header=None)
        threshold=5
    elif args.dataset=='gowalla':
        df = pd.read_csv("raw_data/Gowalla_totalCheckins.txt", sep="\t")
        threshold=15
    else:
        raise Exception("Dataset not recognized. Datasets available: gowalla, ml1m")
        
    if args.threshold:
        threshold = args.threshold
    
    print(threshold)
    
    preprocess(df, args.dataset, threshold, train_size=args.train_size)
