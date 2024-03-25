import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import yaml
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.autograd import Variable
import shap
import torch.cuda.amp as amp
from utils.fairness import demographic_parity, loss_func

def train_cfg_load():
    train_cfg_path = 'config/train.yaml'
    train_cfg_dict = {}
    with open(rf'{train_cfg_path}') as f:
        train_loader = yaml.load(f, Loader=yaml.FullLoader)
        train_cfg_dict['batch_size'] = train_loader['batch_size']
        train_cfg_dict['epochs'] = train_loader['epochs']
        train_cfg_dict['learning_rate'] = train_loader['learning_rate']
        train_cfg_dict['weight_decay_parameter'] = train_loader['weight_decay_parameter']
        train_cfg_dict['momentum'] = train_loader['momentum']
        train_cfg_dict['dampening'] = train_loader['dampening']
        train_cfg_dict['tolerance1'] = train_loader['tolerance1']
        train_cfg_dict['tolerance2'] = train_loader['tolerance2']
    return train_cfg_dict


def save_checkpoint(model, info, place, op):
    weight_save_path = os.path.join('checkpoints', f'{info["model_name"]}_{info["coef"]}_{place}_{op}.pth.tar')
    torch.save(model.state_dict(), weight_save_path)
    print(f'Saved model to {weight_save_path} with min val dp {info["val_dp"]} with val loss {info["val_loss"]}')

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model


def train(train_data, valid_data, model, coef, device, lr, batch_size, place, epochs=0, op='reg'):

    if not batch_size:
        batch_size = train_config['batch_size']
    if not lr:
        lr = train_config['learning_rate']
    
    if epochs == 0:
        epochs = train_config['epochs']
    
    # train loader
    train_config = train_cfg_load()
    X_train_tensor = torch.Tensor(train_data[:, 2:-1].astype(float)).to(device)
    y_train_tensor = torch.Tensor(train_data[:, -1].astype(float)).to(device)
    if op == 'cls':
        y_train_tensor = y_train_tensor.to(torch.long)
    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # val loader
    X_val_tensor = torch.Tensor(valid_data[:, 2:-1].astype(float)).to(device)
    y_val_tensor = torch.Tensor(valid_data[:, -1].astype(float)).to(device)
    if op == 'cls':
        y_val_tensor = y_val_tensor.to(torch.long)
    val_set = TensorDataset(X_val_tensor, y_val_tensor)
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    optim_sgd = optim.SGD(list(model.parameters()), lr=lr,
                                                    weight_decay=train_config['weight_decay_parameter'],
                                                    momentum=train_config['momentum'],
                                                    dampening=train_config['dampening'])
    optim_adam = optim.Adam(list(model.parameters()), lr=lr,
                                                    weight_decay=train_config['weight_decay_parameter'])
    optimizer = optim_adam
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

    tolerance1 = train_config['tolerance1']
    tolerance2 = train_config['tolerance2']

    train_loss_per_epoch = []
    train_dp_per_epoch = []

    val_loss_per_epoch = []
    val_dp_per_epoch = []

    
    best_epoch = 0
    scaler = amp.GradScaler()
    for epoch in tqdm(range(epochs), desc='Training'):
        # training
        model.train()
        train_loss_per_batch = []
        train_dp_per_batch = []

        for _, (X_train, y_train) in enumerate(train_data_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            
            with amp.autocast():
                y_train_pred = model(X_train)
                z_train = X_train[:, -1].to(torch.int)

                dp = demographic_parity(y_train, y_train_pred, z_train, tolerence=tolerance1, op=op)
                loss = loss_func(y_train, y_train_pred, op) + coef * dp / batch_size

            train_loss_per_batch.append(loss.item())
            train_dp_per_batch.append(dp)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        train_loss_this_epoch = sum(train_loss_per_batch) / len(train_loss_per_batch)
        train_dp_this_epoch = sum(train_dp_per_batch) / len(train_dp_per_batch)

        # validation
        model.eval()
        val_loss_per_batch = []
        val_dp_per_batch = []
        with torch.no_grad():
            for _, (X_val, y_val) in enumerate(val_data_loader):
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                with amp.autocast():
                    y_val_pred = model(X_val)
                    z_val = X_val[:, -1].to(torch.int)

                    val_dp = demographic_parity(y_val, y_val_pred, z_val, tolerence=tolerance1, op=op)
                    val_loss = loss_func(y_val, y_val_pred, op) + coef * val_dp / batch_size

                val_loss_per_batch.append(val_loss.item())
                val_dp_per_batch.append(val_dp)

        val_loss_this_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
        val_dp_this_epoch = sum(val_dp_per_batch) / len(val_dp_per_batch)

        
        info = {
            'model_name': model.name,
            'coef': coef,
            'epoch': epoch,
            'train_loss': train_loss_this_epoch,
            'train_dp': train_dp_this_epoch,
            'val_loss': val_loss_this_epoch,
            'val_dp': val_dp_this_epoch,
        }
        
        if coef == 0:
            if epoch == 0 or (epoch>0 and val_loss_this_epoch<min(val_loss_per_epoch)):
                best_epoch = epoch
                save_checkpoint(model, info, place, op)
        else:
            if epoch == 0 or (epoch>0 and val_dp_this_epoch<min(val_dp_per_epoch)):
                best_epoch = epoch
                save_checkpoint(model, info, place, op)
            
        
        train_loss_per_epoch.append(train_loss_this_epoch)
        train_dp_per_epoch.append(train_dp_this_epoch)
        val_loss_per_epoch.append(val_loss_this_epoch)
        val_dp_per_epoch.append(val_dp_this_epoch)

        
        prev_lr = optimizer.param_groups[0]['lr']
        if coef > 0:
            scheduler.step(val_dp_this_epoch)
        else:
            scheduler.step(val_loss_this_epoch)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"Learning rate updated: {prev_lr:.6f} -> {current_lr:.6f}")
    
    train_loss_per_epoch = np.array(train_loss_per_epoch).reshape((-1, 1))
    train_dp_per_epoch = np.array(train_dp_per_epoch).reshape((-1, 1))

    val_loss_per_epoch = np.array(val_loss_per_epoch).reshape((-1, 1))
    val_dp_per_epoch = np.array(val_dp_per_epoch).reshape((-1, 1))

    train_history = np.concatenate((train_loss_per_epoch, train_dp_per_epoch,
                                    val_loss_per_epoch, val_dp_per_epoch), axis=1)
    train_history = np.expand_dims(train_history, axis=0)
    coef = str(coef)
    out_path = f'outputs/{place}_{model.name}_{op}_dict.pkl'
    if not os.path.exists(out_path):
        out_dict = {coef: {
            'train_his': train_history
        }}
    else:
        with open(out_path, 'rb') as f:
            out_dict = pickle.load(f)
        if not coef in out_dict.keys():
            out_dict[coef] = {}
            out_dict[coef]['train_his'] = train_history
        else:
            tmp = out_dict[coef]['train_his']
            out_dict[coef]['train_his'] = np.concatenate([tmp, train_history], axis=0)
    with open(out_path, 'wb') as f:
        pickle.dump(out_dict, f)

    return model





    

