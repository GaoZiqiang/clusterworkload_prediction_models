import torch
import torch.nn as nn
import torch.utils.data as Data

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from model import *
from utils import *


def train(data_pth, resource, batch_size, epochs, save_pth):
    # 得到数据
    x, y = get_train_data(data_pth, resource)
    test_data = x[:20]
    y_real = y[:20]
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好?)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    # embed()
    # 建模三件套：loss，优化，epochs
    model = LstmAutoEncoder()  # lstm
    # model = LstmFcAutoEncoder()  # lstm+fc模型
    loss_function = nn.MSELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    # epochs = 140
    # 开始训练
    model.train()
    for i in range(epochs):
        print("===>Epoch: ", i + 1)
        for seq, labels in train_loader:
            # print("in train(), seq.shape = ", seq.shape)
            # if seq.shape[0] != 20:
            #     continue
            optimizer.zero_grad()
            # embed()
            y_pred = model(seq)  # 压缩维度：得到输出，并将维度为1的去除
            # print("in train(), y_pred.shape = ", y_pred.shape)
            y_pred = y_pred[:, -1, :]# 中间维度只取最后一个的
            # y_pred = model(seq)
            # embed()
            single_loss = loss_function(y_pred.squeeze(), labels)# 将y_pred由[20,1]转换为[20]
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
            # print("Train Step:", i, " loss: ", single_loss)
        # 每20次，输出一次前20个的结果，对比一下效果
        if (i + 1) % 1 == 0:
            # embed()
            y_pred = model(test_data)  # 压缩维度：得到输出，并将维度为1的去除
            y_pred = y_pred[:, -1, :]
            # print("Train epoch: ", i)
            # print("TEST: ", test_data)
            # print("PRED: ", y_pred)
            # print("LOSS: ", loss_function(y_pred, y_real))
            # from IPython import embed
            # embed()
            # test_data_ = test_data.detach().numpy()
            y_pred_ = y_pred.detach().numpy()

            # print("nn.MSELoss= ", loss_function(y_pred.squeeze(), labels))
            print('MSE Value= ', mean_squared_error(y_pred_, y_real))
            # print('MSE2 Value= ', mean_squared_error(y_real, y_pred_))
            print('MAE Value= ', mean_absolute_error(y_pred_, y_real))
            print('RMSE Value= ', sqrt(mean_squared_error(y_pred_, y_real)))
            print('MAPE Value= ', mean_absolute_percentage_error(y_real, y_pred_))

        # Visualising the results
        # if i == epochs - 1:
        #     plt.plot(test_data_, color='red', label='real cpu util percent')
        #     plt.plot(y_pred_, color='green', label='predicted cpu util percent')
        #     plt.title('cpu util percent prediction')
        #     plt.xlabel('Time')
        #     plt.ylabel('cpu util percent')
        #     plt.legend()
        #     plt.show()

    torch.save(model, save_pth)

if __name__ == '__main__':
    ### train
    data_pth = '../data/machine_usage.csv'
    resource = 'cpu'
    batch_size = 20
    epochs = 1
    save_pth = '../output/' + 'slidwin60_hidden_size16_' + resource + '_epoch_' + str(epochs) + '(CNN).pth'

    train(data_pth, resource, batch_size, epochs, save_pth)