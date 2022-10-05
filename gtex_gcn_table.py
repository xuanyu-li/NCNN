import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import time
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
def get_te(layer, n_hidden, hidden_layer_conv):
    in_size = 943
    n_hidden = n_hidden
    out_size = 9520
    learning_rate = 3e-4
    init_vals = np.sqrt(6.0 / (np.array([in_size, n_hidden, n_hidden]) + np.array([n_hidden, n_hidden, out_size])))
    n_epoch = 300
    n_batch = 200
    decay_factor = 0.9
    drop_rate = 0.1
    hidden_channels = hidden_layer_conv # gcn hidden channel


    ### generate co-expression graph
    X_tr=np.load('GTEx_X_tr_float64.npy')
    X_te=np.load('GTEx_X_te_float64.npy')
    X_va=np.load('GTEx_X_va_float64.npy')
    Y_tr=np.load('GTEx_Y_tr_float64.npy')
    Y_te=np.load('GTEx_Y_te_float64.npy')
    Y_va=np.load('GTEx_Y_va_float64.npy')



    threshold = 0.5
    thresholdgraph = 137
    corrmat = torch.load('stringDB1thres' + str(thresholdgraph) +'.pth')
    print(f'this is the result for threshold{thresholdgraph}')
    corrmat = corrmat + torch.diag(torch.ones(in_size))
    adjancency = (corrmat != 0)



    adjancent = adjancency.long()
    adjancentsum = adjancent.sum(1)

    aloneindex = torch.where(adjancentsum==1)[0]
    alonenum = (adjancentsum == 1).count_nonzero().item()
    conectindex = torch.where(adjancentsum !=1)[0]
    conectnum = (adjancentsum != 1).count_nonzero().item()


    connectadjacent = adjancency[conectindex]
    connectadjacent = connectadjacent.transpose(0,1)[conectindex]



    # connectadjacent = connectadjacent.numpy()


    adjancency = adjancency.numpy()
    corrmat = torch.tensor(adjancency)
    # threshold = 0.5
    # corrmat = torch.load(str(threshold)+'corrroboust.pth')
    #corrmat = torch.tensor(corrmat).float()
    # print(np.count_nonzero(corrmat))
    # print(f'total number {942*942}')
    # def count_non(threshold):
    #     corrmat = dataframe.corr()
    #     corrmat = np.array(corrmat)
    #
    #     corrmat = corrmat - np.identity(943)
    #     corrmat[corrmat < threshold] = 0
    #     print(f'threshold{threshold}')
    #     print(np.count_nonzero(corrmat))
    #
    # count_non(0.6)
    # count_non(0.7)
    X_tr = torch.from_numpy(X_tr).float()
    Y_tr = torch.from_numpy(Y_tr).float()
    X_va = torch.from_numpy(X_va).float()
    Y_va = torch.from_numpy(Y_va).float()
    X_te = torch.from_numpy(X_te).float()
    Y_te = torch.from_numpy(Y_te).float()

    X_tr = X_tr.cuda()
    Y_tr = Y_tr.cuda()
    X_va = X_va.cuda()
    Y_va = Y_va.cuda()
    X_te = X_te.cuda()
    Y_te = Y_te.cuda()

    coomat, sideweight = dense_to_sparse(corrmat)
    coomat = coomat.cuda()
    sideweight = sideweight.cuda()
    data_list = []
    print(f'graph generated')
    for i in range(X_tr.shape[0]):
        data = Data(x=torch.unsqueeze(X_tr[i, :], 1), y=torch.unsqueeze(Y_tr[i, :], 0),
                    edge_attr=sideweight, edge_index=coomat)
        data_list.append(data)
    loader = DataLoader(data_list, batch_size= n_batch)
    valid_list = []
    test_list = []
    print(f'train data loaded')
    for i in range(X_va.shape[0]):
        data = Data(x=torch.unsqueeze(X_va[i, :], 1), y=torch.unsqueeze(Y_va[i, :], 0),
                    edge_attr=sideweight, edge_index=coomat)
        valid_list.append(data)
    valid_loader = DataLoader(valid_list, batch_size=n_batch)
    print(f'valid data loaded')
    for i in range(X_te.shape[0]):
        data = Data(x=torch.unsqueeze(X_te[i, :], 1), y=torch.unsqueeze(Y_te[i, :], 0),
                    edge_attr=sideweight, edge_index=coomat)
        test_list.append(data)
    test_loader = DataLoader(test_list, batch_size=n_batch)
    print(f'test data loaded')

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(1, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels, out_size)
        def forward(self, x, edge_index, batch):
            # 1. Obtain node embeddings
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)
            # 2. Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin(x)

            return x

    if layer == 3:
        class GCN_1layer_fc(torch.nn.Module):
            def __init__(self, hidden_channels, n_hidden):
                super(GCN_1layer_fc, self).__init__()
                torch.manual_seed(12345)
                # self.conv1 = GINConv(
                #     nn.Sequential(
                #         Linear(1, hidden_channels),
                #         nn.BatchNorm1d(hidden_channels),
                #         nn.ReLU(),
                #         Linear(hidden_channels, hidden_channels),
                #         nn.BatchNorm1d(hidden_channels),
                #         nn.ReLU(),
                #     ), train_eps=False)
                self.conv1 = GraphConv(1, hidden_channels)
                self.linear1 = Linear(in_size*hidden_channels,n_hidden)
                self.linear2 = Linear(n_hidden, n_hidden)

               # self.linear3 = Linear(n_hidden,n_hidden)
                self.lin = Linear(n_hidden, out_size)

            def forward(self, x, edge_index):
                # 1. Obtain node embeddings
                x = self.conv1(x, edge_index)
                x = x.relu()

                # 2. Readout layer flatten all the tensors
                x = x.reshape(-1, in_size*hidden_channels)

                # 3. fully-connected layer
                x = self.linear1(x)
                x = x.tanh()
                x = self.linear2(x)
                x = x.tanh()

               # x = self.linear3(x)
               # x = x.tanh()

                # 4. Apply a final classifier
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.lin(x)

                return x

    if layer == 2:
        class GCN_1layer_fc(torch.nn.Module):
            def __init__(self, hidden_channels, n_hidden):
                super(GCN_1layer_fc, self).__init__()
                torch.manual_seed(12345)
                # self.conv1 = GINConv(
                #     nn.Sequential(
                #         Linear(1, hidden_channels),
                #         nn.BatchNorm1d(hidden_channels),
                #         nn.ReLU(),
                #         Linear(hidden_channels, hidden_channels),
                #         nn.BatchNorm1d(hidden_channels),
                #         nn.ReLU(),
                #     ), train_eps=False)
                self.conv1 = GraphConv(1, hidden_channels)
                self.linear1 = Linear(in_size * hidden_channels, n_hidden)


                # self.linear3 = Linear(n_hidden,n_hidden)
                self.lin = Linear(n_hidden, out_size)

            def forward(self, x, edge_index):
                # 1. Obtain node embeddings
                x = self.conv1(x, edge_index)
                x = x.relu()

                # 2. Readout layer flatten all the tensors
                x = x.reshape(-1, in_size * hidden_channels)

                # 3. fully-connected layer
                x = self.linear1(x)
                x = x.tanh()
                # 4. Apply a final classifier
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.lin(x)

                return x

    if layer == 1:
        class GCN_1layer_fc(torch.nn.Module):
            def __init__(self, hidden_channels, n_hidden):
                super(GCN_1layer_fc, self).__init__()
                torch.manual_seed(12345)
                # self.conv1 = GINConv(
                #     nn.Sequential(
                #         Linear(1, hidden_channels),
                #         nn.BatchNorm1d(hidden_channels),
                #         nn.ReLU(),
                #         Linear(hidden_channels, hidden_channels),
                #         nn.BatchNorm1d(hidden_channels),
                #         nn.ReLU(),
                #     ), train_eps=False)
                self.conv1 = GraphConv(1, hidden_channels)
                self.linear1 = Linear(in_size * hidden_channels, out_size)

            def forward(self, x, edge_index):
                # 1. Obtain node embeddings
                x = self.conv1(x, edge_index)
                x = x.relu()

                # 2. Readout layer flatten all the tensors
                x = x.reshape(-1, in_size * hidden_channels)

                # 3. fully-connected layer
                x = self.linear1(x)

                return x

    model = GCN_1layer_fc(hidden_channels, n_hidden)
    for para in model.parameters():
        init.uniform_(para, -0.0001, 0.0001)

    model.cuda()
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer,
            'min', factor=decay_factor, threshold=1e-6, patience=0, min_lr=1e-5)
    train_ls, valid_mae, test_mae = [], [], []
    MAE_va_best = 10.0
    def test(loader):
        model.eval()
        loss = 0
        for data in loader:  # Iterate in batches over the validation/test dataset.
            out = model(data.x, data.edge_index)
            loss += torch.abs(out - data.y).sum().item()

        return loss / (len(loader.dataset)*out_size)  # Derive ratio of correct predictions.

    for epoch in range(n_epoch):
        model.train()
        t_old = time.time()
        for data in loader:
            ## predict
            out = model(data.x, data.edge_index)
            loss = loss_f(data.y, out)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # train_loss = evaluate_mae(dataset, MLP)
        # MAE_va = evaluate_mae(datasetvalid, MLP)
        # MAE_te = evaluate_mae(datasettest, MLP)
        ###此处由于内存不足的原因，可能需要使用batch计算mae
        with torch.no_grad():
            train_loss = test(loader)
            MAE_va = test(valid_loader)
            MAE_te = test(test_loader)
            train_ls.append(train_loss)
            valid_mae.append(MAE_va)
            test_mae.append(MAE_te)

            if MAE_va < MAE_va_best:
                MAE_va_best = MAE_va
                torch.save(model.state_dict(), 'H3_model_gcn.pt')
            t_new = time.time()

            if epoch % 5 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch : {epoch} Training Loss: {train_loss:.5f} MAE_va:{MAE_va:.5f} \
        MAE_test:{MAE_te:.5f} training time: {t_new - t_old} lr: {curr_lr:.9f}')

            lr_scheduler.step(MAE_va)   # 这里或许可以使用validloss
            #lr_scheduler.step()


    #
    # MLP.load_state_dict(torch.load('H3_model_drop.pt'))
    # MAE_va = torch.abs(y_va - MLP(x_va)).mean().item()
    # MAE_te = torch.abs(y_te - MLP(x_te)).mean().item()
    # print(f'best valid mae:{MAE_va} record:{MAE_va_best} best test mae: {MAE_te}')

    model.load_state_dict(torch.load('H3_model_gcn.pt'))
    N = X_te.shape[0]
    with torch.no_grad():
        MAE_te = test(test_loader)
        loss = 0
        for data in test_loader:  # Iterate in batches over the validation/test dataset.
            out = model(data.x, data.edge_index)
            loss += torch.abs(out - data.y).sum(dim=0)
        SD_te = torch.var(loss / N)
        SD_te = math.sqrt(SD_te)

    return  MAE_te, SD_te


hidden_layers = [1, 2, 3]
hidden_nodes = [3000, 6000, 9000]
conv = [4, 7, 11]


table = np.zeros([3,3])
dataframe = pd.DataFrame(table,
                   columns=['3000', '6000', '9000'])

for i in range(0,3):
    for j in range(0,3):
        mae, sd = get_te(hidden_layers[i], hidden_nodes[j], conv[j])
        dataframe.iloc[i, j] = str(mae)[:6]+'+'+str(sd)[:6]
        # table_va[i, j] = sd


print(dataframe.to_latex())
dataframe.to_csv('Gtex_gcn.csv')
