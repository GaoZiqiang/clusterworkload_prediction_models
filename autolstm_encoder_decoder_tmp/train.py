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

import matplotlib.pyplot as plt

from model import *
from utils import *


def train(data_pth, resource, batch_size, epochs, save_pth):
    # for validate
    predictions = []
    reals = []

    # 指标
    min_MAPE = 1
    min_MAPE_epoch = -1
    # 得到数据
    x, y = get_train_data(data_pth, resource)
    test_data = x[0:1040]
    y_real = y[1000:1040]

    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好?)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(test_data, y_real),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
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

        # validate
        for seq, labels in test_loader:
            y_pred = model(seq)  # 压缩维度：得到输出，并将维度为1的去除
            y_pred = y_pred[:, -1, :]  # 取window_size个预测值中的最后一个值
            y_pred_ = y_pred.detach().numpy()

            # embed()
            predictions.extend(y_pred_.squeeze())
            reals.extend(labels.detach().numpy())

            # print("nn.MSELoss= ", loss_function(y_pred.squeeze(), labels))
        MSE = mean_squared_error(predictions, reals)

        print('MSE Value= ', MSE)
        # print('MSE2 Value= ', mean_squared_error(y_real, y_pred_))
        print('MAE Value= ', mean_absolute_error(predictions, reals))
        print('RMSE Value= ', sqrt(mean_squared_error(predictions, reals)))
        MAPE = mean_absolute_percentage_error(reals, predictions)
        print('MAPE Value= ', MAPE)

        if min_MAPE > MAPE:
            min_MAPE = MAPE
            min_MAPE_epoch = i + 1

        if MAPE <= 0.065:
            print("------------ get a proper model ------------")
            torch.save(model, "../figures/another_" + str(MAPE) + "_epoch" + str(i + 1) + ".pth")

        # Visualising the results
        # if i == epochs - 1:
        #     plt.plot(test_data_, color='red', label='real cpu util percent')
        #     plt.plot(y_pred_, color='green', label='predicted cpu util percent')
        #     plt.title('cpu util percent prediction')
        #     plt.xlabel('Time')
        #     plt.ylabel('cpu util percent')
        #     plt.legend()
        #     plt.show()

    ### save model
    # torch.save(model, save_pth)
    print("min_MAPE = ", min_MAPE)
    print("min_MAPE_epoch = ", min_MAPE_epoch)
    torch.save(model, "../figures/" + "FINAL.pth")
    ### Visualising the losses
    # plt.plot(total_MSE, color='red', label='MSE loss')
    # plt.plot(total_MAPE, color='green', label='MAPE loss')
    # plt.title('training losses')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    #
    # epoch = [x for x in range(len(total_MSE))]
    #
    # for a, b in zip(epoch, total_MSE):
    #     plt.text(a, b, b, ha="center", va="bottom", fontsize=10)
    #
    # for a, b in zip(epoch, total_MAPE):
    #     plt.text(a, b, b, ha="center", va="bottom", fontsize=10)
    #
    # plt.legend()
    # plt.show()

    # torch.save(model, save_pth)

if __name__ == '__main__':
    ### train
    data_pth = '../data/machine_usage.csv'
    resource = 'cpu'
    batch_size = 20
    epochs = 4000
    save_pth = '../output/' + 'utils' + resource + '_epoch_' + str(epochs) + '.pth'

    train(data_pth, resource, batch_size, epochs, save_pth)