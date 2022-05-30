import torch
import torch.nn as nn
import torch.utils.data as Data

import time
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


def test(data_pth, resource, test_len, model_pth, batch_size):
    # 定义损失参数
    total_MSE = 0
    total_MAE = 0
    total_RMSE = 0
    total_MAPE = 0
    min_MAPE = 1

    # 加载模型
    # model_pth = "../output/auto_encoder_epoch140.pth"
    test_model = torch.load(model_pth,map_location = torch.device('cpu'))
    test_model.eval()

    # 加载数据
    x, y = get_train_data(data_pth, resource)
    test_data = x[:test_len]
    y_real = y[:test_len]
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(test_data, y_real),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )

    for inputs, targets in test_loader:
        y_pred = test_model(inputs)  # 压缩维度：得到输出，并将维度为1的去除
        y_pred = y_pred[:, -1, :]
        y_pred_ = y_pred.detach().numpy()
        targets_ = targets.detach().numpy()

        MSE = mean_squared_error(y_pred_, targets_)
        MAE = mean_absolute_error(y_pred_, targets_)
        RMSE = sqrt(mean_squared_error(y_pred_, targets_))
        MAPE = mean_absolute_percentage_error(targets_, y_pred_)
        min_MAPE = min(min_MAPE, MAPE)

        # print('MSE Value= ', MSE)
        # print('MAE Value= ', MAE)
        # print('RMSE Value= ', RMSE)
        # print('MAPE Value= ', MAPE)

        total_MSE += MSE
        total_MAE += MAE
        total_RMSE += RMSE
        total_MAPE += MAPE

    loader_len = len(test_loader)
    # embed()
    test_MSE = total_MSE / loader_len
    test_MAE = total_MAE / loader_len
    test_RMSE = total_RMSE / loader_len
    test_MAPE = total_MAPE / loader_len
    print("min_MAPE = ", min_MAPE)

    return test_MSE, test_MAE, test_RMSE, test_MAPE


if __name__ == '__main__':
    ### test
    data_pth = '../data/machine_usage.csv'
    resource = 'cpu'
    test_len = 240
    model_name = 'MAPE0.04895193_epoch299.pth'
    model_pth = '../output/' + model_name
    batch_size = 20

    MSE, MAE, RMSE, MAPE = test(data_pth, resource, test_len, model_pth, batch_size)
    print(MSE, MAE, RMSE, MAPE)
    with open('../result_in_txt/results_train.txt', 'a') as file:
        file.write(
            'Date %s | model_name = %s | MSE = %.5f | MAE: %.5f | RMSE: %.5f | MAPE: %.5f |\n' % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), model_name, MSE, MAE, RMSE, MAPE))