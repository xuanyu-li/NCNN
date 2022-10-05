import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import time
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import math
torch.manual_seed(2)

def get_te(layer, n_hidden, hidden_layer_conv):
    in_size = 943
    n_hidden = n_hidden
    out_size = 9520
    learning_rate = 3e-4
    init_vals = np.sqrt(6.0 / (np.array([in_size, n_hidden, n_hidden]) + np.array([n_hidden, n_hidden, out_size])))
    n_epoch = 400
    n_batch = 200
    decay_factor = 0.9
    drop_rate = 0.15
    hidden_layer_conv = hidden_layer_conv

    # read in the data
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
    # adjancency = adjancency.numpy()
    if layer ==4:
        MLP = nn.Sequential(
            nn.Linear(conectnum * hidden_layer_conv + alonenum, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, out_size),
        )

    if layer == 3:
        MLP = nn.Sequential(
            # nn.Linear(in_size, n_hidden),
            # nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.Linear(conectnum*hidden_layer_conv+alonenum, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, out_size),
        )

    if layer == 2:
        MLP = nn.Sequential(
            # nn.Linear(in_size, n_hidden),
            # nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.Linear(conectnum * hidden_layer_conv + alonenum, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, out_size),
        )

    if layer == 1:
        MLP = nn.Sequential(
            # nn.Linear(in_size, n_hidden),
            # nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.Linear(conectnum * hidden_layer_conv + alonenum, out_size),
        )




    ## 需要转为numpy
    class graph_individual_conv(torch.nn.Module):
        def __init__(self, in_size, n_hidden, hidden_layer_conv):
            super(graph_individual_conv, self).__init__()
            self.adjancency = adjancency
            names = self.__dict__
            for i in range(conectnum):
                size = torch.count_nonzero(connectadjacent.long()[i])
                names['weight'+str(i)] = nn.Linear(size, hidden_layer_conv)
            #self.bn = nn.BatchNorm1d(in_size * hidden_layer_conv)
            self.mlp = MLP
        def forward(self, x):
            names = self.__dict__
            x1 = x[:,self.adjancency[conectindex[0]]]
            y = self.weight0(x1)
            for i in range(1, conectnum):
                x1 = x[:, self.adjancency[conectindex[i]]]
                linear = names['weight'+str(i)]
                tmp = linear(x1)
                y = torch.cat([y, tmp], dim=1)
            x_alone = x[:, aloneindex]
            y = torch.cat([y, x_alone], dim=1)
            y = y.tanh()
            # y = self.bn(y)
            # y = torch.nn.functional.dropout(y, p=drop_rate, training=self.training)
            y = MLP(y)
            return y

    model = graph_individual_conv(in_size,n_hidden, hidden_layer_conv)


    # initialization with uniform distribution
    for para in model.parameters():
        # init.uniform_(para, -init_vals[0], init_vals[0])
        if len(para.shape) != 1:
            nn.init.xavier_uniform_(para)

    init.uniform_(model.mlp[-1].weight, -0.0001, 0.0001)


    #  define loss , optimizer
    loss_f = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=decay_factor, threshold=1e-7, patience=0, min_lr=1e-5)
    #lr_scheduler = StepLR(optimizer, step_size=20, gamma = decay_factor)
    # convert data into tensor
    x_tr = torch.from_numpy(X_tr).float()
    y_tr = torch.from_numpy(Y_tr).float()
    x_va = torch.from_numpy(X_va).float()
    y_va = torch.from_numpy(Y_va).float()
    x_te = torch.from_numpy(X_te).float()
    y_te = torch.from_numpy(Y_te).float()

    ### min-max minimization
    # y = torch.cat([y_tr, y_va], dim=1)
    #
    # y_max, id = y.max()
    # y_min, id = y.min()
    #
    # x = (x-x_min)/(x_max - x_min)
    # y = (y-y_min)/(y_max - y_min)
    #
    # x_tr = x[:88807, :]
    # x_va = x[88807:99908, :]
    # x_te = x[99908:, :]
    #
    # y_tr = y[:88807, :]
    # y_va = y[88807:99908, :]
    # y_te = y[99908:, :]

    ### put data and model into GPU
    if torch.cuda.is_available():
        x_tr = x_tr.cuda()
        y_tr = y_tr.cuda()
        x_va = x_va.cuda()
        y_va = y_va.cuda()
        x_te = x_te.cuda()
        y_te = y_te.cuda()
        model = model.cuda()

    for i in range(conectnum):
        exec('model.weight{}.cuda()'.format(i))

    ### define dataloader
    dataset = Data.TensorDataset(x_tr, y_tr)

    data_iter  = Data.DataLoader(dataset=dataset, batch_size=n_batch, shuffle=True)
    # valid_iter = Data.DataLoader(dataset=datasetvalid, batch_size=n_batch, shuffle=False)
    # test_iter  = Data.DataLoader(dataset=datasettest, batch_size=n_batch, shuffle=False)

    train_ls, valid_mae, test_mae = [], [], []
    # def evaluate_mae(data_iter, model):
    #     acc_sum, n = 0.0, 0
    #     model.eval()
    #     for X,y in data_iter:
    #         acc_sum += torch.abs(y - model(X)).sum().item()
    #         model.train()
    #         n += y.shape[0]
    #     model.train()
    #     return acc_sum / n
    MAE_va_best = 10.0
    #print(f'scale{y_max - y_min}')
    ### training process
    for epoch in range(n_epoch):
        model.train()
        t_old = time.time()
        for x_tr, y_tr in data_iter:
            ## predict
            y_hat = model(x_tr)
            loss = loss_f(y_tr, y_hat)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # train_loss = evaluate_mae(dataset, model)
        # MAE_va = evaluate_mae(datasetvalid, model)
        # MAE_te = evaluate_mae(datasettest, model)
        # with torch.no_grad():
        #     if epoch % 5 == 0:
        #         Rob_va = torch.abs(model(x_va) - model(x_va)).mean().item()
        #         Rob_te = torch.abs(model(x_te) - model(x_te)).mean().item()
        #         Rob_tr = torch.abs(model(x_tr) - model(x_tr)).mean().item()
        #
        #         print(f'Epoch : {epoch} Training rob Loss: {Rob_tr:.5f} Rob_va:{Rob_va:.5f} \
        #             Rob_test:{Rob_te:.5f} ')



        model.eval()
        ###此处由于内存不足的原因，可能需要使用batch计算mae
        with torch.no_grad():
            train_loss = torch.abs(model(x_tr) - y_tr).mean().item()
            if epoch % 5 == 0:
                MAE_va = torch.abs(y_va - model(x_va)).mean().item()   ### 注意tensor 和 numpy 之间的转换
                MAE_te = torch.abs(y_te - model(x_te)).mean().item()


                train_ls.append(train_loss)
                valid_mae.append(MAE_va)
                test_mae.append(MAE_te)

                if MAE_va < MAE_va_best:
                    MAE_va_best = MAE_va
                    torch.save(model.state_dict(), 'H3_gtex_sparse_mlp_model_drop.pt')
                t_new = time.time()


                curr_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch : {epoch} Training Loss: {train_loss:.5f} MAE_va:{MAE_va:.5f} \
        MAE_test:{MAE_te:.5f} training time: {t_new - t_old} lr: {curr_lr:.9f}')


            lr_scheduler.step(train_loss)   # 这里或许可以使用validloss
            #lr_scheduler.step()

    train_ls  = np.array(torch.tensor(train_ls).cpu())
    valid_mae = np.array(torch.tensor(valid_mae).cpu())
    test_mae  = np.array(torch.tensor(test_mae).cpu())

    np.save('H3_train.npy', train_ls)
    np.save('H3_valid.npy', valid_mae)
    np.save('H3_test.npy', test_mae)

    model.load_state_dict(torch.load('H3_gtex_sparse_mlp_model_drop.pt'))
    model.eval()
    with torch.no_grad():
        MAE_va = torch.abs(y_va - model(x_va)).mean().item()
        MAE_te = torch.abs(y_te - model(x_te)).mean().item()
        SD_te = torch.var(torch.mean(torch.abs(y_te - model(x_te)), dim=0)).item()
        SD_te = math.sqrt(SD_te)    

    print(f'best valid mae:{MAE_va} record:{MAE_va_best} best test mae: {MAE_te}')

    plt.figure(figsize=(10,5))
    plt.title("Three mae curves")
    plt.plot(train_ls,label="train")
    plt.plot(valid_mae, label = "valid")
    plt.plot(test_mae,label="test")
    plt.yscale("log")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("threecurve.jpg")

    return MAE_te, SD_te



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
dataframe.to_csv('GTEx_sparse.csv')
