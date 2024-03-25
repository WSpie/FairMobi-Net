import warnings
warnings.filterwarnings('ignore')

import os, sys, math
import numpy as np
from argparse import ArgumentParser
import time
from sklearn.model_selection import train_test_split
import torch
import sys
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import yaml
from pathlib import Path

from log import Log
from models.fairmb import FairMB_BINARY, FairMB_REG
# from models.dg import DG
# from models.gmel import GMEL
# from models.rm import rm_main
# from models.rt import TransformerModel
from train import *
from test import *
from preprocessing import *
from utils.fairness import *
from utils.metrics import *

log_train = Log('train')
log_test = Log('test')

def choose_model(model_name, op):
    model_name = model_name.lower()
    func_flag = False
    if model_name == 'fairmb':
        if op == 'reg':
            return FairMB_REG, func_flag
        elif op == 'binary':
            return FairMB_BINARY, func_flag
        else:
            raise Exception(f'No such option: {op}')
    # elif model_name == 'dg':
    #     return DG, func_flag
    # elif model_name == 'gmel':
    #     return GMEL, func_flag
    # elif model_name == 'rt':
    #     return TransformerModel, func_flag
    # elif model_name == 'rm':
    #     func_flag = True
    #     return rm_main, func_flag
    else:
        raise Exception(f'No such model: {model_name}!')

if __name__ == '__main__':
    cfg = yaml.safe_load(Path('condif/train.yaml'))
    parser = ArgumentParser()
    parser.add_argument('--round', type=int, default=0, help='use the index of row in orders to keep the data identically')
    parser.add_argument('--device', type=str, default='cpu', help='cuda|cpu')
    parser.add_argument('--place', type=str, default='Atlanta')
    parser.add_argument('--model', type=str, default='fairmb', help='model name abbrivation')
    parser.add_argument('--coef', type=float, default=0, help='must be non-negative')
    parser.add_argument('--lr', type=float, default=cfg['learning_rate'])
    parser.add_argument('--batch-size',type=int, default=cfg['batch_size'])
    parser.add_argument('--test-batch-size', type=int, default=512)
    parser.add_argument('--op', type=str, default='reg')
    parser.add_argument('--epochs', type=int, default=cfg['epochs'])
    opt = parser.parse_args()

    try:
        # check gpu
        if opt.device == 'cuda' and not torch.cuda.is_available():
            print(f'{opt.device} is not available!')
            opt.device = 'cpu'
        device = torch.device(opt.device)
        print(device.type)
        # check model availablity
        MODEL, func_flag = choose_model(opt.model, opt.op)
    except Exception as e:
        print(e)
        sys.exit(0)
    
    data_path = f'data/processed/{opt.place}_feat.parquet'
    data_df = pd.read_parquet(data_path)
    data_df = data_df[data_df['count'] <= 50]

    train_df, val_df = train_test_split(data_df, test_size=0.4, random_state=0)
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
    
    train_data = train_df.to_numpy()
    val_data = val_df.to_numpy()

    if not func_flag:
        # except radiation model
        model = MODEL(train_data.shape[1] - 3, dim_out=1).to(device)

        try:
            print('Start training')
            model = train(train_data, val_data, model, opt.coef, device, opt.lr, opt.batch_size, opt.place, opt.epochs, op=opt.op)
        except Exception as e:
            print('Train error', e)
            log_train.exception()
            sys.exit(0)

        try:
            print('Start testing')
            test(test_df, model, opt.coef, device, opt.place, opt.test_batch_size, op=opt.op)
        except Exception as e:
            print('Test error', e)
            log_test.exception()
            sys.exit(0)
        print('Finished')

    else:
        result_df = MODEL(data_df, pd.read_csv(f'data/processed/{opt.place}_poly.csv'))



