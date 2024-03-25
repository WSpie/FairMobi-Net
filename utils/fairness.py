import numpy as np
import torch
import torch.nn as nn
from itertools import combinations

def r2_score(y_true, y_pred):
    # y_pred = torch.clamp(y_pred, min=0)
    y_pred = torch.abs(y_pred)
    y_pred = torch.ceil(y_pred)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.to('cpu').detach().numpy()


def mae_loss(y_true_tensor, y_pred_tensor):
    return torch.mean(torch.abs(y_true_tensor - y_pred_tensor))

def mse_loss(y_true_tensor, y_pred_tensor):
    return torch.mean((y_true_tensor - y_pred_tensor)**2)

def rmse_loss(y_true_tensor, y_pred_tensor):
    return torch.sqrt(torch.mean((y_true_tensor - y_pred_tensor)**2))


def loss_func(y_true_tensor, y_pred_tensor, op='reg', use='mae'):
    use = use.lower()
    if op == 'cls':
        cls_loss = nn.CrossEntropyLoss()
        return cls_loss(y_pred_tensor, y_true_tensor)
    else:
        if use == 'mae':
            return mae_loss(y_true_tensor, y_pred_tensor)
        elif use == 'mse':
            return mse_loss(y_true_tensor, y_pred_tensor)
        elif use == 'rmse':
            return rmse_loss(y_true_tensor, y_pred_tensor)

    
def disparate_impact_regression_multi(y_pred_tensor, z_tensor):
    pairwise_ratios = []
    
    z_class, z_counts = z_tensor.unique(return_counts=True)
    for comb in combinations(z_class, 2):
        z0, z1 = comb[0].item(), comb[1].item()
        idx0 = (z_tensor==z0).nonzero(as_tuple=True)[0]
        idx1 = (z_tensor==z1).nonzero(as_tuple=True)[0]
        ratio = torch.mean(y_pred_tensor[idx0]) / (torch.mean(y_pred_tensor[idx1]) + 1e-6)
        pairwise_ratios.append(ratio.to('cpu').detach().numpy())

    return pairwise_ratios

def compute_fairness_metrics(y_true_tensor, y_pred_tensor, z_tensor, op='reg'):
    # MAE ratio
    # statistic parity
    # disparity impact
    
    z_tensor = z_tensor.to(torch.int)
    z_cls = torch.sort(torch.unique(z_tensor)).values
    
    if op == 'reg':
        mae_overall = mae_loss(y_true_tensor, y_pred_tensor)
        mae_ratios = [] # -> 1 the better
        stat_parities = [] # -> lower the better

        
        for cls in z_cls:
            cls_indices = torch.nonzero(z_tensor == cls).squeeze()
            y_true_cls = y_true_tensor[cls_indices]
            y_pred_cls = y_pred_tensor[cls_indices]
            
            mae_cls = mae_loss(y_true_cls, y_pred_cls)
            mae_ratios.append((mae_cls / (mae_overall+1e-6)).to('cpu').detach().numpy())
            stat_parity_cls = (torch.abs(torch.mean(y_pred_cls) - torch.mean(y_pred_tensor))).to('cpu').detach().numpy()
            stat_parities.append(stat_parity_cls)
            
        dis_impacts = disparate_impact_regression_multi(y_pred_tensor, z_tensor)
        
        fair_metrics = np.array([
                np.mean(mae_ratios),
                np.mean(stat_parities),
                np.mean(dis_impacts)
                ]).reshape((1, -1))
        
        return fair_metrics
    
    elif op == 'cls':
        y_pred_tensor = torch.argmax(y_pred_tensor, dim=1)
        num_classes = torch.unique(y_true_tensor).size(0)
        equal_opportunities = []
        equalized_odds = []
        dis_impacts = []

        for comb in combinations(z_cls, 2):
            z0, z1 = comb[0].item(), comb[1].item()
            idx0 = (z_tensor==z0).nonzero(as_tuple=True)[0]
            idx1 = (z_tensor==z1).nonzero(as_tuple=True)[0]

            equal_opportunity_per_class = []
            equalized_odds_per_class = []
            dis_impact_per_class = []

            for class_idx in range(num_classes):
                # Equal Opportunity
                tpr0 = torch.sum((y_pred_tensor[idx0] == class_idx) & (y_true_tensor[idx0] == class_idx)) / torch.sum(y_true_tensor[idx0] == class_idx)
                tpr1 = torch.sum((y_pred_tensor[idx1] == class_idx) & (y_true_tensor[idx1] == class_idx)) / torch.sum(y_true_tensor[idx1] == class_idx)
                equal_opportunity_per_class.append(torch.abs(tpr0 - tpr1).to('cpu').detach().numpy())

                # Equalized Odds
                fpr0 = torch.sum((y_pred_tensor[idx0] == class_idx) & (y_true_tensor[idx0] != class_idx)) / torch.sum(y_true_tensor[idx0] != class_idx)
                fpr1 = torch.sum((y_pred_tensor[idx1] == class_idx) & (y_true_tensor[idx1] != class_idx)) / torch.sum(y_true_tensor[idx1] != class_idx)
                equalized_odds_per_class.append(torch.abs(fpr0 - fpr1).to('cpu').detach().numpy())

                # Disparate Impact
                ratio0 = torch.sum(y_pred_tensor[idx0] == class_idx) / len(idx0)
                ratio1 = torch.sum(y_pred_tensor[idx1] == class_idx) / len(idx1)
                dis_impact_per_class.append((ratio0 / (ratio1 + 1e-6)).to('cpu').detach().numpy())

            equal_opportunities.append(np.mean(equal_opportunity_per_class))
            equalized_odds.append(np.mean(equalized_odds_per_class))
            dis_impacts.append(np.mean(dis_impact_per_class))

        fair_metrics = np.array([
            np.mean(equal_opportunities),
            np.mean(equalized_odds),
            np.mean(dis_impacts)
        ]).reshape((1, -1))

        return fair_metrics
        
        
def demographic_parity(y_true_tensor, y_pred_tensor, z_tensor, tolerence=0.05, weight=False, op='reg'):
    dp = 0
    mean_loss = loss_func(y_true_tensor, y_pred_tensor, op)
    z_tensor = z_tensor.to(torch.int)
    z_class, z_counts = z_tensor.unique(return_counts=True)
    
    if op == 'reg':
        for comb in combinations(z_class, 2):
            z0, z1 = comb[0].item(), comb[1].item()
            idx0 = (z_tensor==z0).nonzero(as_tuple=True)[0]
            idx1 = (z_tensor==z1).nonzero(as_tuple=True)[0]
            dp0 = torch.nonzero(torch.abs(y_true_tensor[idx0]-y_pred_tensor[idx0]) <= mean_loss+tolerence).size()[0] / (z_counts[z0]+1e-10)
            dp1 = torch.nonzero(torch.abs(y_true_tensor[idx1]-y_pred_tensor[idx1]) <= mean_loss+tolerence).size()[0] / (z_counts[z1]+1e-10)
            if weight:
                w = y_true_tensor.shape[0] / (idx0.shape[0] + idx1.shape[0])
            else:
                w = 1
            dp += w * torch.abs(dp0 - dp1)

    if op == 'cls':
        # Calculate accuracy for all samples
        y_pred_tensor = torch.argmax(y_pred_tensor, dim=1)
        acc_all = (y_true_tensor == y_pred_tensor).sum().float() / y_true_tensor.shape[0]

        # Calculate accuracy for each group
        normalize = 0
        count = 0
        for z_value in z_class:
            count += 1
            group_indices = (z_tensor == z_value).nonzero(as_tuple=True)[0]
            acc_gn = (y_true_tensor[group_indices] == y_pred_tensor[group_indices]).sum().float() / z_counts[z_value]
            if weight:
                w = y_true_tensor.shape[0] / z_counts[z_value]
                normalize += w
            else:
                w = 1
            dp += w * torch.abs(acc_gn - acc_all)
        dp /= count
    if type(dp) == int:
        return dp
    else:
        return dp.item()
    
