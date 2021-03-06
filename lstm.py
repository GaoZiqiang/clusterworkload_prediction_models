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
import time

from IPython import embed



def get_train_data(path):
    # Importing the training set
    dataset_train = pd.read_csv(path, error_bad_lines=False, sep="\t")
    training_set = dataset_train.iloc[0:3400, 2:3].values

    # normalization
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # sliding window
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(120, 3320):
        X_train.append(training_set_scaled[i - 120:i, 0])
        y_train.append(training_set_scaled[[i,i+1,i+2,i+4,i+5,i+6], 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series)

    return get_tensor_from_pd(X_train).float(), get_tensor_from_pd(y_train).float()



class LstmAutoEncoder(nn.Module):
    def __init__(self, input_layer=120, hidden_layer=100, output_layer=6, batch_size=20):
        super(LstmAutoEncoder, self).__init__()

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_layer, self.hidden_layer, batch_first=True)
        self.fc = nn.Linear(self.hidden_layer, self.output_layer)

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        # lstm
        lstm, (n, c) = self.lstm(input_x,
                                 (torch.zeros(1, self.batch_size, self.hidden_layer),
                                 torch.zeros(1, self.batch_size, self.hidden_layer)))
        output = self.fc(lstm)
        # embed()

        return output.squeeze()



def train(epochs):
    min_MAPE = 1
    min_MAPE_epoch = -1
    # ????????????
    x, y = get_train_data('./data/machine_usage.csv')
    test_data = x[:20]
    y_real = y[:20]
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # ?????????Data.TensorDataset()????????????????????????????????????
        batch_size=20,  # ???????????????
        shuffle=False,  # ????????????????????? (???????????????)
        num_workers=2,  # ????????????multiprocess???????????????
    )

    # ??????????????????loss????????????epochs
    model = LstmAutoEncoder()  # lstm
    # model = LstmFcAutoEncoder()  # lstm+fc??????
    loss_function = nn.MSELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ?????????
    # epochs = 1
    # ????????????
    model.train()
    for i in range(epochs):
        print("===>Epoch: ", i+1)
        for seq, labels in train_loader:
            optimizer.zero_grad()
            # embed()
            y_pred = model(seq).squeeze()  # ?????????????????????????????????????????????1?????????
            # y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            # ???????????????????????????????????????????????????????????????????????????print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
            # print("Train Step:", i, " loss: ", single_loss)
        # ???20?????????????????????20?????????????????????????????????
        if (i + 1) % 1 == 0:
            # embed()
            y_pred = model(test_data).squeeze()  # ?????????????????????????????????????????????1?????????
            # print("Train epoch: ", i)
            # print("TEST: ", test_data)
            # print("PRED: ", y_pred)
            # print("LOSS: ", loss_function(y_pred, y_real))
            # from IPython import embed
            # embed()
            # test_data_ = test_data.detach().numpy()
            y_pred_ = y_pred.detach().numpy()

            MSE = mean_squared_error(y_pred_, y_real)
            MAE = mean_absolute_error(y_pred_, y_real)
            RMSE = sqrt(mean_squared_error(y_pred_, y_real))
            MAPE = mean_absolute_percentage_error(y_pred_, y_real)
            if MAPE < min_MAPE:
                min_MAPE = MAPE
                min_MAPE_epoch = i + 1
            # min_MAPE = min(min_MAPE, MAPE)
            # min_MAPE_epoch = i + 1
            print('MSE Value= ', MSE)
            print('MAE Value= ', MAE)
            print('RMSE Value= ', RMSE)
            print('MAPE Value= ', MAPE)
            if MAPE <= 0.09:
                print("------------ A Proper Model ------------")
            with open('./result_in_txt/lstm_trainlog.txt', 'a') as file:
                file.write(
                    'epoch=%s | Date=%s | MSE = %.7f | MAE: %.7f | RMSE: %.7f | MAPE: %.7f |\n' % (
                        str(i+1), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), MSE, MAE, RMSE, MAPE))

    print("min_MAPE = ", min_MAPE)
    print("epoch = ", min_MAPE_epoch)
    # torch.save(model, './output/' + 'auto_encoder_' +'MAPE' + str(min_MAPE) +'_epoch' + str(epochs) + '.pth')

if __name__ == '__main__':
    epoches = 800
    train(epoches)

