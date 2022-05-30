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

from IPython import embed

def data_plt(path, resource):
    dataset_train = pd.read_csv(path, error_bad_lines=False, sep="\t")
    # embed()
    training_set = []
    if resource == "cpu":
        training_set = dataset_train.iloc[51294:71295, [2]].values# 使用cpu disk预测cpu
    elif resource == "mem":
        training_set = dataset_train.iloc[0:3325, 3:4].values
    elif resource == "disk":
        training_set = dataset_train.iloc[0:3325, 8:9].values
    elif resource == "net":
        training_set = dataset_train.iloc[0:3325, 4:5].values
    else:
        print("unknown resouce type, exit")
        return

    ### Visualising the losses
    plt.plot(training_set, color='grey', label='MSE loss')
    # plt.plot(training_set, color='green', label='MAPE loss')
    plt.title('training losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def get_train_data2(path, resource):
    # Importing the training set
    dataset_train = pd.read_csv(path, error_bad_lines=False, sep="\t")
    # embed()
    training_set = []
    if resource == "cpu":
        training_set = dataset_train.iloc[0:10517, [2,8]].values# 使用cpu disk预测cpu
    elif resource == "mem":
        training_set = dataset_train.iloc[0:3325, 3:4].values
    elif resource == "disk":
        training_set = dataset_train.iloc[0:3325, 8:9].values
    elif resource == "net":
        training_set = dataset_train.iloc[0:3325, 4:5].values
    else:
        print("unknown resouce type, exit")
        return

    # normalization
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # training_set_scaled = training_set
    new = []
    for i in range(1,len(training_set_scaled)):
        new.append([training_set_scaled[i-1][0], training_set_scaled[i-1][1], training_set_scaled[i][1],training_set_scaled[i][0]])

    new = np.reshape(new, (len(new), -1))
    # embed()


    # embed()
    # sliding window
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    # sliding window = 120
    # sliding window = 60
    for i in range(120, 1000):
        X_train.append(new[i - 120:i, 0:3])# 输入为120*num_features
        y_train.append(new[i, -1])# 输出为1*1
    X_train, y_train = np.array(X_train), np.array(y_train)
    # embed()
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], -1, X_train.shape[1]))

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series)

    # embed()
    return get_tensor_from_pd(X_train).float(), get_tensor_from_pd(y_train).float()


def get_train_data(path, resource):
    # Importing the training set
    dataset_train = pd.read_csv(path, error_bad_lines=False, sep="\t")
    # embed()
    training_set = []
    if resource == "cpu":
        training_set = dataset_train.iloc[0:11000, [2,8]].values# 使用cpu disk预测cpu [2,3,8]cpu+mem+disk [2,6,7,8]cpu+neti+neto+disk
    elif resource == "mem":
        training_set = dataset_train.iloc[0:3325, 3:4].values
    elif resource == "disk":
        training_set = dataset_train.iloc[0:3325, 8:9].values
    elif resource == "net":
        training_set = dataset_train.iloc[0:3325, 4:5].values
    else:
        print("unknown resouce type, exit")
        return

    # normalization
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # embed()
    # sliding window
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    # sliding window = 120
    # sliding window = 60
    # 3320 10000 19380
    for i in range(120, 10000):# 3320
        X_train.append(training_set_scaled[i - 120:i, :])# 输入为120*num_features
        ### 通过改变y的长度实现不同的prediction step
        y_train.append(training_set_scaled[[i,i+1], 0])# 输出为1*1 0:target为cpu 1:target为disk
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], -1, X_train.shape[1]))

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series)

    # embed()
    return get_tensor_from_pd(X_train).float(), get_tensor_from_pd(y_train).float()


if __name__ == "__main__":
    # get_train_data("../data/machine_usage.csv", "cpu")
    # X, y = get_train_data2("../data/machine_usage.csv", "cpu")
    # print(X.shape, y.shape)
    data_plt('../data/machine_usage.csv', 'cpu')
