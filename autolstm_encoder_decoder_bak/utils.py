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


def get_train_data():
    """得到训练数据，这里使用随机数生成训练数据，由此导致最终结果并不好"""

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series.values)

    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    # 生成训练数据x并做归一化后，构造成dataframe格式，再转换为tensor格式
    df = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(np.random.randint(0, 10, size=(2000, 300))))
    y = pd.Series(np.random.randint(0, 2, 2000))
    return get_tensor_from_pd(df).float(), get_tensor_from_pd(y).float()

def get_train_data(path, resource):
    # Importing the training set
    dataset_train = pd.read_csv(path, error_bad_lines=False, sep="\t")
    training_set = []
    if resource == "cpu":
        training_set = dataset_train.iloc[0:3325, 2:3].values / 100
    elif resource == "mem":
        training_set = dataset_train.iloc[0:3325, 3:4].values / 100
    elif resource == "disk":
        training_set = dataset_train.iloc[0:3325, 8:9].values / 100
    elif resource == "net":
        training_set = dataset_train.iloc[0:3325, 4:5].values / 100
    else:
        print("unknown resouce type, exit")
        return

    # normalization
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # sliding window
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(120, 3320):
        X_train.append(training_set_scaled[i - 120:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series)

    return get_tensor_from_pd(X_train).float(), get_tensor_from_pd(y_train).float()
