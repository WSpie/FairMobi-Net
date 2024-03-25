from models import Basic
import torch
import torch.nn as nn

class FairMB(Basic):
    def __init__(self, dim_input, dim_hidden=64, droupout_p=0.5, sub_feat_nums=20):
        super(Basic, self).__init__()
        self.name = 'MMLP'
        self.dim_imput_i = self.dim_imput_j = sub_feat_nums
        self.dim_imput_share = dim_input - self.dim_imput_i - self.dim_imput_j

        self.linear_i1 = nn.Linear(self.dim_imput_i, 2*dim_hidden)
        self.relu_i1 = nn.GELU()
        self.bn_i1 = nn.BatchNorm1d(2*dim_hidden)
        self.droupout_i1 = nn.Dropout(droupout_p)
        self.linear_i2 = nn.Linear(2*dim_hidden, 2*dim_hidden)
        self.relu_i2 = nn.GELU()
        self.bn_i2 = nn.BatchNorm1d(2*dim_hidden)
        self.droupout_i2 = nn.Dropout(droupout_p)

        self.linear_j1 = nn.Linear(self.dim_imput_j, 2*dim_hidden)
        self.relu_j1 = nn.GELU()
        self.bn_j1 = nn.BatchNorm1d(2*dim_hidden)
        self.droupout_j1 = nn.Dropout(droupout_p)
        self.linear_j2 = nn.Linear(2*dim_hidden, 2*dim_hidden)
        self.relu_j2 = nn.GELU()
        self.bn_j2 = nn.BatchNorm1d(2*dim_hidden)
        self.droupout_j2 = nn.Dropout(droupout_p)

        self.linear_share = nn.Linear(self.dim_imput_share, dim_hidden // 4)
        self.relu_share = nn.GELU()
        self.droupout_share = nn.Dropout(droupout_p)

        self.relu_all0 = nn.GELU()
        self.linear_all1 = nn.Linear(2*dim_hidden+dim_hidden//4, 4*dim_hidden)
        self.relu_all1 = nn.GELU()
        self.bn_all1 = nn.BatchNorm1d(4*dim_hidden)
        self.droupout_all1 = nn.Dropout(droupout_p)
        self.linear_all2 = nn.Linear(4*dim_hidden, 4*dim_hidden)
        self.relu_all2 = nn.GELU()
        self.bn_all2 = nn.BatchNorm1d(4*dim_hidden)
        self.droupout_all2 = nn.Dropout(droupout_p)
        self.linear_all3 = nn.Linear(4*dim_hidden, 2*dim_hidden)
        self.relu_all3 = nn.GELU()
        self.bn_all3 = nn.BatchNorm1d(2*dim_hidden)
        self.droupout_all3 = nn.Dropout(droupout_p)
        self.linear_all4 = nn.Linear(2*dim_hidden, 2*dim_hidden)
        self.relu_all4 = nn.GELU()
        self.bn_all4 = nn.BatchNorm1d(2*dim_hidden)
        self.droupout_all4 = nn.Dropout(droupout_p)
        self.linear_all5 = nn.Linear(2*dim_hidden, dim_hidden)
        self.relu_all5 = nn.GELU()
        self.bn_all5 = nn.BatchNorm1d(dim_hidden)
        self.droupout_all5 = nn.Dropout(droupout_p)

        # self.outputs = nn.Linear(dim_hidden, 1)
    
    def forward(self, X):

        X_i = X[:, :self.dim_imput_i]
        X_j = X[:, self.dim_imput_i:self.dim_imput_i+self.dim_imput_j]
        X_share = X[:, self.dim_imput_i+self.dim_imput_j:]
        
        X_i = self.linear_i1(X_i)
        X_i = self.relu_i1(X_i)
        X_i = self.droupout_i1(X_i)
        X_i = self.bn_i1(X_i)
        X_i = self.linear_i2(X_i)
        X_i = self.relu_i2(X_i)
        X_i = self.droupout_i2(X_i)
        X_i = self.bn_i2(X_i)

        X_j = self.linear_j1(X_j)
        X_j = self.relu_j1(X_j)
        X_j = self.droupout_j1(X_j)
        X_j = self.bn_j1(X_j)
        X_j = self.linear_j2(X_j)
        X_j = self.relu_j2(X_j)
        X_j = self.droupout_j2(X_j)
        X_j = self.bn_j2(X_j)

        X_ij = X_i + X_j
        
        X_share = self.linear_share(X_share)
        X_share = self.relu_share(X_share)
        X_share = self.droupout_share(X_share)

        X = torch.cat([X_ij, X_share], dim=1)
        X = self.relu_all0(X)

        X = self.linear_all1(X)
        X = self.relu_all1(X)
        X = self.droupout_all1(X)
        X = self.bn_all1(X)
        X = self.linear_all2(X)
        X = self.relu_all2(X)
        X = self.droupout_all2(X)
        X = self.bn_all2(X)
        X = self.linear_all3(X)
        X = self.relu_all3(X)
        X = self.droupout_all3(X)
        X = self.bn_all3(X)
        X = self.linear_all4(X)
        X = self.relu_all4(X)
        X = self.droupout_all4(X)
        X = self.bn_all4(X)
        X = self.linear_all5(X)
        X = self.relu_all5(X)
        X = self.droupout_all5(X)
        X = self.bn_all5(X)

        # X = self.outputs(X)
        # X = nn.ReLU()(X)
        # X = torch.abs(X)
        # X = torch.floor(X)
        return X

class FairMB_REG(FairMB):
    def __init__(self, dim_input, dim_hidden=64, dropout_p=0.5, dim_output=1, sub_feat_nums=20):
        super().__init__(dim_input, dim_hidden, dropout_p, sub_feat_nums)
        self.outputs = nn.Linear(dim_hidden, dim_output)
    def forward(self, X):
        X = super().forward(X)
        X = self.outputs(X)
        return X
    
class FairMB_BINARY(FairMB):
    def __init__(self, dim_input, dim_hidden=128, dropout_p=0.35, dim_output=1, sub_feat_nums=20):
        super().__init__(dim_input, dim_hidden, dropout_p, sub_feat_nums)
        self.outputs = nn.Linear(dim_hidden, dim_output)
    def forward(self, X):
        X = super().forward(X)
        X = self.outputs(X)
        X = nn.Sigmoid()(X)
        return X