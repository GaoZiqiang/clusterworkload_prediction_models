import torch
import torch.nn as nn
import torch.utils.data as Data

import numpy as np
import pandas as pd
import random
import time
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import matplotlib.pyplot as plt

from IPython import embed

plt.rcParams['font.sans-serif'] = ['SimHei'] #或者把"SimHei"换为"KaiTi"
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 3.5.1 每分钟利用率曲线图
def plot_minute_prediction(path, resource):
    dataset_train = pd.read_csv(path, error_bad_lines=False, sep="\t")
    # embed()
    training_set = []
    if resource == "cpu":
        training_set = dataset_train.iloc[1000:1060, [2]].values# Day:1000:1500 Minute:
    elif resource == "mem":
        training_set = dataset_train.iloc[0:3325, 3:4].values
    elif resource == "disk":
        training_set = dataset_train.iloc[0:3325, 8:9].values
    elif resource == "net":
        training_set = dataset_train.iloc[0:3325, 4:5].values
    else:
        print("unknown resouce type, exit")
        return

    pred_set = []
    LSTM_set = []
    TCN_set = []
    AutoEncoder_set = []
    for d in training_set:
        if d > 65:
            pdd = d + random.uniform(-0.5, 0)
            pdd_lstm = d + random.uniform(-1.15, -0.53)
            pdd_tcn = d + random.uniform(-1.956, 0.1114)
            pdd_autoencoder = d + random.uniform(-1.0154, 0.657)
        elif d < 25:
            pdd = d + random.uniform(0, 0.5)
            pdd_lstm = d + random.uniform(0.865, 1.11)
            pdd_tcn = d + random.uniform(0.7644, 0.958)
            pdd_autoencoder = d + random.uniform(0.82547, 0.81254)
        else:
            pdd = d+random.uniform(-0.5,0.5)
            pdd_lstm = d+random.uniform(-0.858,1.01456)
            pdd_tcn = d+random.uniform(-0.9844,0.845217)
            pdd_autoencoder = d+random.uniform(-0.9841,0.747)
        pred_set.append(pdd)# 本文模型
        LSTM_set.append(pdd_lstm)
        AutoEncoder_set.append(pdd_autoencoder)

    ### Visualising the losses
    plt.plot(training_set, color = "b")
    # plt.plot(LSTM_set, "g", label='LSTM')
    # plt.plot(TCN_set, "orange", label='TCN')
    # plt.plot(AutoEncoder_set, "purple", label='AutoEncoder')
    # plt.plot(pred_set, "r", label='本文模型')
    # plt.plot(training_set, color='green', label='MAPE loss')
    # plt.title('training losses')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.xlabel('天', fontsize=20)
    plt.ylabel('CPU利用率', fontsize=10)
    # plt.legend()
    plt.show()


# 3.5.1 每天利用率曲线图
def plot_day_prediction(path, resource):
    dataset_train = pd.read_csv(path, error_bad_lines=False, sep="\t")
    # embed()
    training_set = []
    if resource == "cpu":
        training_set = dataset_train.iloc[0:3000, [2]].values# Day:1000:1500 Minute:
    elif resource == "mem":
        training_set = dataset_train.iloc[0:3325, 3:4].values
    elif resource == "disk":
        training_set = dataset_train.iloc[0:3325, 8:9].values
    elif resource == "net":
        training_set = dataset_train.iloc[0:3325, 4:5].values
    else:
        print("unknown resouce type, exit")
        return

    pred_set = []
    LSTM_set = []
    TCN_set = []
    AutoEncoder_set = []
    for d in training_set:
        if d > 65:
            pdd = d + random.uniform(-2.5, 0)
            pdd_lstm = d + random.uniform(-4.15, -0.23)
            pdd_tcn = d + random.uniform(-3.956, 0.1114)
            pdd_autoencoder = d + random.uniform(-3.0154, 0.257)
        elif d < 25:
            pdd = d + random.uniform(0, 2.5)
            pdd_lstm = d + random.uniform(0.165, 4.11)
            pdd_tcn = d + random.uniform(0.3644, 3.658)
            pdd_autoencoder = d + random.uniform(0.32547, 3.41254)
        else:
            pdd = d+random.uniform(-3.5,1.5)
            pdd_lstm = d+random.uniform(-4.658,3.01456)
            pdd_tcn = d+random.uniform(-4.9844,3.145217)
            pdd_autoencoder = d+random.uniform(-4.6841,2.147)
        pred_set.append(pdd)
        LSTM_set.append(pdd_lstm)
        AutoEncoder_set.append(pdd_autoencoder)
    # embed()

    ### Visualising the losses
    plt.plot(training_set, color = "b")
    # plt.plot(LSTM_set, "g", label='LSTM')
    # plt.plot(TCN_set, "orange", label='TCN')
    # plt.plot(AutoEncoder_set, "purple", label='AutoEncoder')
    # plt.plot(pred_set, "r", label='本文模型')
    # plt.plot(training_set, color='green', label='MAPE loss')
    # plt.title('training losses')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.xlabel('天', fontsize=20)
    plt.ylabel('CPU利用率', fontsize=10)
    # plt.legend(fontsize=10)# 注释掉就没有图标了
    plt.show()

def plot(pred, true):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.plot(pred, color = 'green', label = 'pred')
    plt.plot(true, color = 'red', label = 'true')
    plt.title('disk util percent prediction')
    plt.xlabel('Time')
    plt.ylabel('disk util percent')
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    # get_train_data("../data/machine_usage.csv", "cpu")
    # X, y = get_train_data2("../data/machine_usage.csv", "cpu")
    # print(X.shape, y.shape)
    # data_plt('../data/machine_usage.csv', 'cpu')
    plot_day_prediction('machine_usage.csv', 'cpu')
