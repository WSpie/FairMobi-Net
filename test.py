import torch
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from utils.fairness import demographic_parity, loss_func, compute_fairness_metrics
from train import load_checkpoint
import torch_geometric
from torch_geometric.data import Data
import math
from preprocessing import *

def test(test_df, model, coef, device, place, batch_size=256, pred=False, op='reg'):
    tolerance1 = 0
    tolerance2 = 0
    coef = np.round_(coef, 1)
    
    # load model
    checkpoint_path = f'checkpoints/{model.name}_{coef}_{place}_{op}.pth.tar'
    model = load_checkpoint(checkpoint_path, model)

    test_data = test_df.to_numpy()
    # test loader

    X_test_tensor = torch.Tensor(test_data[:, 2:-1].astype(float)).to(device)
    y_test_tensor = torch.Tensor(test_data[:, -1].astype(float)).to(device)
    if op == 'cls':
        y_test_tensor = y_test_tensor.to(torch.long)
    test_set = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model.eval()
    test_losses = []
    test_dps = []
    test_fair_metrics = np.array([])
    preds = np.array([])

    for _, (X_test, y_test) in enumerate(test_loader):
        X_test = X_test.to(device)
        y_test =y_test.to(device)

        y_pred = model(X_test)
        z_test = X_test[:, -1].to(torch.int)

        test_dp = demographic_parity(y_test, y_pred, z_test, tolerence=tolerance1, op=op)
        test_loss = loss_func(y_test, y_pred, op) + coef * test_dp / batch_size
        
        test_fair = compute_fairness_metrics(y_test, y_pred, z_test, op)
        test_losses.append(test_loss.item())
        test_dps.append(test_dp)
        if not np.any(test_fair_metrics):
            test_fair_metrics = test_fair
        else:
            test_fair_metrics = np.vstack((test_fair_metrics, test_fair))
        if op == 'cls':
            y_pred = torch.argmax(y_pred, dim=1)
        preds = np.append(preds, y_pred.to('cpu').detach().numpy())
            
    if op == 'reg':
        preds = np.maximum(0, preds)
        preds = np.round_(preds)

    test_overall_loss = sum(test_losses) / len(test_losses)
    test_overall_dp = sum(test_dps) / len(test_dps)
    test_overall_fair_metrics = np.mean(test_fair_metrics, axis=0).reshape((1, -1))
    test_history = np.array([test_overall_loss, test_overall_dp]).reshape((1, -1))
    test_history = np.hstack((test_history, test_overall_fair_metrics))
    coef = str(coef)
    out_path = f'outputs/{place}_{model.name}_{op}_dict.pkl'
    with open(out_path, 'rb') as f:
        out_dict = pickle.load(f)
    if 'test_dict' not in out_dict[coef].keys():
        out_dict[coef]['test_dict'] = {}
        
    if 'test_res' not in out_dict[coef]['test_dict'].keys():
        out_dict[coef]['test_dict']['test_res'] = test_history
        out_dict[coef]['test_dict']['test_df'] = test_df
    else:
        test_res_tmp = out_dict[coef]['test_dict']['test_res']
        out_dict[coef]['test_dict']['test_res'] = np.vstack((test_res_tmp, test_history))
        
    pred_cnt = len([col for col in out_dict[coef]['test_dict']['test_df'].columns if col.startswith('pred')])
    out_dict[coef]['test_dict']['test_df'][f'pred{pred_cnt}'] = preds
    
    with open(out_path, 'wb') as f:
        pickle.dump(out_dict, f)

def generate_loader(device, test_batch_size=256):
    places = ['Atlanta', 'Harris', 'King', 'Suffolk']
    train_dfs = []
    train_loaders = []
    test_dfs = []
    test_features_list = []
    test_loaders = []
    for place in places:
        train_df, test_df, train_mask, val_mask, feature_cols, feat_scales = load_df(place)
        train_features = train_df[feature_cols].values / feat_scales
        train_features_tensor = torch.tensor(train_features, dtype=torch.float).to(device)
        train_labels = train_df['count'].values
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float).to(device)
        train_data = Data(x=train_features_tensor, y=train_labels_tensor)
        train_data.train_mask = train_mask
        train_data.val_mask = val_mask
        train_loader = torch_geometric.data.DataLoader([train_data], batch_size=2048, shuffle=True)
        train_loaders.append(train_loader)
        
        # test_df = pd.concat([train_df.iloc[val_mask], test_df], axis=0).reset_index(drop=True)
        test_features = test_df[feature_cols].astype(float).values / feat_scales
        test_features_tensor = torch.tensor(test_features, dtype=torch.float).to(device)
        test_labels = test_df['count'].values
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)
        test_set = TensorDataset(test_features_tensor, test_labels_tensor)
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
        test_loaders.append(test_loader)
        # test_dfs.append(test_df)
        
        # print(len(train_df.iloc[val_mask]), len(train_df))
        test_dfs.append(test_df)
        train_dfs.append(train_df)
        test_features_list.append(test_features)
    return train_dfs, train_loaders, test_dfs, test_features_list, test_loaders, feature_cols, feat_scales

def prediction(reg_model, loader, loss_func, device, binary_model=None, rdn_flag=False):
    reg_model.eval()
    trues = torch.tensor([], dtype=torch.int)
    preds = torch.tensor([], dtype=torch.int)
    dps = []
    for _, (X_test, y_test) in enumerate(loader):
        X_test = X_test.to(device)
        y_test = y_test.to(device).unsqueeze(1)
        with torch.no_grad():
            reg_pred = reg_model(X_test)
            pred = reg_pred
            if rdn_flag:
                pred += torch.randn(pred.shape).to(device) * 0.8
            pred = torch.abs(torch.round(pred))
            if binary_model:
                pred = torch.clamp(pred, min=1)
                binary_model.eval()
                binary_pred = binary_model(X_test)
                binary_pred = torch.round(binary_pred)
                pred *= binary_pred
            # pred = torch.round(pred)
            z = X_test[:, -3:]
            loss = loss_func(pred, y_test)
            dp = demographic_parity(y_test, pred, z, loss)
            dps.append(dp)
            trues = torch.cat([trues, y_test.squeeze().to('cpu')])
            preds = torch.cat([preds, pred.squeeze().to('cpu')])
    trues, preds = trues.numpy(), preds.numpy()
    dp_avg = np.mean(dps)
    return trues, preds, [dp_avg]