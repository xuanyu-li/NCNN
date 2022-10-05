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


def get_te(layer, n_hidden):
    in_size = 943
    n_hidden = n_hidden
    out_size = 9520
    learning_rate = 3e-4
    init_vals = np.sqrt(6.0 / (np.array([in_size, n_hidden, n_hidden]) + np.array([n_hidden, n_hidden, out_size])))
    n_epoch = 200
    n_batch = 200
    decay_factor = 0.9
    drop_rate = 0.15

    # read in the data
    X_tr=np.load('GTEx_X_tr_float64.npy')
    X_te=np.load('GTEx_X_te_float64.npy')
    X_va=np.load('GTEx_X_va_float64.npy')
    Y_tr=np.load('GTEx_Y_tr_float64.npy')
    Y_te=np.load('GTEx_Y_te_float64.npy')
    Y_va=np.load('GTEx_Y_va_float64.npy')
    if layer == 4:
        MLP = nn.Sequential(
            nn.Linear(in_size, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, n_hidden),
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
            nn.Linear(in_size, n_hidden),
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

    if layer == 2:
        MLP = nn.Sequential(
            nn.Linear(in_size, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, out_size),
        )

    if layer == 1:
        MLP = nn.Sequential(
            nn.Linear(in_size, n_hidden),
            nn.Tanh(),
            nn.Dropout(drop_rate),
            nn.Linear(n_hidden, out_size),
        )



    class mlp(torch.nn.Module):
        def __init__(self, in_size, n_hidden, out_size, drop_rate):
            super(mlp, self).__init__()
            self.MLP1 = nn.Sequential(
                nn.Linear(in_size, n_hidden),
                nn.Tanh(),
                nn.Dropout(drop_rate),
            )
            self.MLP2 = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
                nn.Dropout(drop_rate),
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
                nn.Dropout(drop_rate),
                nn.Linear(n_hidden, out_size),
            )
        def forward(self, x):
            y = self.MLP1(x)
            y = self.MLP2(y)
            return y

    def max_norm(model, max_val=3, eps=1e-8):
        for name, param in model.named_parameters():
            if 'bias' not in name:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, 0, max_val)
                param = param * (desired / (eps + norm))


    # MLP = mlp(in_size, n_hidden, out_size, drop_rate)
    # init.uniform_(MLP.MLP1[0].weight,-init_vals[0], init_vals[0])
    # init.uniform_(MLP.MLP1[0].bias  ,-init_vals[0], init_vals[0])
    # init.uniform_(MLP.MLP2[0].weight,-init_vals[1], init_vals[1])
    # init.uniform_(MLP.MLP2[0].bias  ,-init_vals[1], init_vals[1])
    # init.uniform_(MLP.MLP2[3].weight,-init_vals[1], init_vals[1])
    # init.uniform_(MLP.MLP2[3].bias  ,-init_vals[1], init_vals[1])
    # init.uniform_(MLP.MLP2[6].weight, -0.0001, 0.0001)
    # init.uniform_(MLP.MLP2[6].bias, -0.0001, 0.0001)




    #initialization with uniform distribution
    # init.uniform_(MLP[0].weight, -init_vals[0], init_vals[0])
    # init.uniform_(MLP[0].bias, -init_vals[0], init_vals[0])
    # init.uniform_(MLP[3].weight, -init_vals[1], init_vals[1])
    # init.uniform_(MLP[3].bias, -init_vals[1], init_vals[1])
    # init.uniform_(MLP[6].weight, -init_vals[1], init_vals[1])
    # init.uniform_(MLP[6].bias, -init_vals[1], init_vals[1])
    # init.uniform_(MLP[9].weight, -0.0001, 0.0001)
    # init.uniform_(MLP[9].bias, -0.0001, 0.0001)



    #initialization with uniform distribution
    # init.uniform_(MLP[0].weight, -init_vals[0], init_vals[0])
    # init.uniform_(MLP[0].bias, -init_vals[0], init_vals[0])
    # init.uniform_(MLP[3].weight, -init_vals[1], init_vals[1])
    # init.uniform_(MLP[3].bias, -init_vals[1], init_vals[1])
    # init.uniform_(MLP[6].weight, -init_vals[1], init_vals[1])
    # init.uniform_(MLP[6].bias, -init_vals[1], init_vals[1])
    # init.uniform_(MLP[9].weight, -0.0001, 0.0001)
    # init.uniform_(MLP[9].bias, -0.0001, 0.0001)
    # init.uniform_(MLP[12].weight, -0.0001, 0.0001)
    # init.uniform_(MLP[12].bias, -0.0001, 0.0001)

    for para in MLP.parameters():
        # init.uniform_(para, -init_vals[0], init_vals[0])
        if len(para.shape) != 1:
            nn.init.xavier_uniform_(para)

    init.uniform_(MLP[-1].weight, -0.0001, 0.0001)


    #  define loss , optimizer
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(MLP.parameters(), lr=learning_rate)
    # optimizer = optim.Adam([{'params': MLP.MLP1.parameters()},
    #                         {'params': MLP.MLP2.parameters(), 'lr':3*learning_rate}
    #                         ], lr=learning_rate)

    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=decay_factor, threshold=1e-6, patience=0, min_lr=1e-5)
    #lr_scheduler = StepLR(optimizer, step_size=20, gamma = decay_factor)
    # convert data into tensor
    x_tr = torch.from_numpy(X_tr).float()
    y_tr = torch.from_numpy(Y_tr).float()
    x_va = torch.from_numpy(X_va).float()
    y_va = torch.from_numpy(Y_va).float()
    x_te = torch.from_numpy(X_te).float()
    y_te = torch.from_numpy(Y_te).float()


    ### put data and model into GPU
    if torch.cuda.is_available():
        x_tr = x_tr.cuda()
        y_tr = y_tr.cuda()
        x_va = x_va.cuda()
        y_va = y_va.cuda()
        x_te = x_te.cuda()
        y_te = y_te.cuda()
        MLP.cuda()

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
    ### training process
    for epoch in range(n_epoch):
        MLP.train()
        t_old = time.time()
        for x_tr, y_tr in data_iter:
            ## predict

            y_hat = MLP(x_tr)
            loss = loss_f(y_tr, y_hat)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #max_norm(MLP)


        # train_loss = evaluate_mae(dataset, MLP)
        # MAE_va = evaluate_mae(datasetvalid, MLP)
        # MAE_te = evaluate_mae(datasettest, MLP)
        MLP.eval()
        ###此处由于内存不足的原因，可能需要使用batch计算mae
        with torch.no_grad():
            MAE_va = torch.abs(y_va - MLP(x_va)).mean().item()   ### 注意tensor 和 numpy 之间的转换
            MAE_te = torch.abs(y_te - MLP(x_te)).mean().item()
            train_loss = torch.abs(MLP(x_tr) - y_tr).mean().item()

            train_ls.append(train_loss)
            valid_mae.append(MAE_va)
            test_mae.append(MAE_te)

            if MAE_va < MAE_va_best:
                MAE_va_best = MAE_va
                torch.save(MLP.state_dict(), 'H3_gtex_model_drop.pt')
            t_new = time.time()

            if epoch % 5 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch : {epoch} Training Loss: {train_loss:.5f} MAE_va:{MAE_va:.5f} \
        MAE_test:{MAE_te:.5f} training time: {t_new - t_old} lr: {curr_lr:.9f}')

            lr_scheduler.step(train_loss)   # 这里或许可以使用validloss
            #lr_scheduler.step()

    train_ls  = np.array(torch.tensor(train_ls).cpu())
    valid_mae = np.array(torch.tensor(valid_mae).cpu())
    test_mae  = np.array(torch.tensor(test_mae).cpu())


    MLP.load_state_dict(torch.load('H3_gtex_model_drop.pt'))
    MLP.eval()
    MAE_va = torch.abs(y_va - MLP(x_va)).mean().item()
    MAE_te = torch.abs(y_te - MLP(x_te)).mean().item()
    SD_te  = torch.var(torch.mean(torch.abs(y_te - MLP(x_te)), dim = 0)).item()
    SD_te = math.sqrt(SD_te)
    print(f'best valid mae:{MAE_va} record:{MAE_va_best} best test mae: {MAE_te}')
    prediction =  MLP(x_te).cpu()
    predict = prediction.detach().numpy()
    np.save('H3_prediction.npy', predict)
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



hidden_layers = [1,2,3]
hidden_nodes = [3000,6000,9000]


table = np.zeros([3,3])
dataframe = pd.DataFrame(table,
                   columns=['3000', '6000', '9000'])
for i in range(0,3):
    for j in range(0,3):
        mae, sd = get_te(hidden_layers[i], hidden_nodes[j])
        dataframe.iloc[i, j] = str(mae)[:6]+'+'+str(sd)[:6]


print(dataframe.to_latex())
dataframe.to_csv('GTEx_mlp.csv')
