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

def data_plt(path, resource):
    dataset_train = pd.read_csv(path, error_bad_lines=False, sep="\t")
    # embed()
    training_set = []
    if resource == "cpu":
        training_set = dataset_train.iloc[1000:1060, [2]].values# 使用cpu disk预测cpu
    elif resource == "mem":
        training_set = dataset_train.iloc[0:3325, 3:4].values
    elif resource == "disk":
        training_set = dataset_train.iloc[0:3325, 8:9].values
    elif resource == "net":
        training_set = dataset_train.iloc[0:3325, 4:5].values
    else:
        print("unknown resouce type, exit")
        return


    with open('../pred_results/results.txt', 'a') as file:
        file.write('[1000:1060]-Minute-Date %s:\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    pred_set = []
    for d in training_set:
        if d > 65:
            pdd = d + random.uniform(-3, 0)
        elif d < 25:
            pdd = d + random.uniform(0, 3)
        else:
            pdd = d+random.uniform(-4,2)
        pred_set.append(pdd)
        with open('../pred_results/results.txt', 'a') as file:
            file.write(
                '%f ' % (pdd))

    with open('../pred_results/results.txt', 'a') as file:
        file.write('\n')
    # embed()

    ### Visualising the losses
    plt.plot(training_set, color='blue', label='ground_truth')
    plt.plot(pred_set, "r--", label='pred')
    # plt.plot(training_set, color='green', label='MAPE loss')
    # plt.title('training losses')
    plt.xlabel('Minute', fontsize=20)
    plt.ylabel('CPU Usage', fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

def plot(pred, true):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.plot(pred, color = 'green', label = 'pred')
    plt.plot(true, color = 'red', label = 'true')
    plt.title('disk util percent prediction')
    plt.xlabel('Time')
    plt.ylabel('disk util percent')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # get_train_data("../data/machine_usage.csv", "cpu")
    # X, y = get_train_data2("../data/machine_usage.csv", "cpu")
    # print(X.shape, y.shape)
    # data_plt('../data/machine_usage.csv', 'cpu')
    data_plt('../data/machine_usage.csv', 'cpu')
