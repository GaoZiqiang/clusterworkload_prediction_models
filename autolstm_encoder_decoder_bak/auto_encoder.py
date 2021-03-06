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
        training_set = dataset_train.iloc[0:3325, 2:3].values / 100
    else:
        print("unknown resouce, return")
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



class LstmAutoEncoder(nn.Module):
    def __init__(self, input_layer=120, hidden_layer=100, output_layer=1, batch_size=20):
        super(LstmAutoEncoder, self).__init__()

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.batch_size = batch_size
        self.encoder_lstm = nn.LSTM(self.input_layer, self.hidden_layer, batch_first=True)
        self.decoder_lstm = nn.LSTM(self.hidden_layer, self.output_layer, batch_first=True)

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        # encoder
        encoder_lstm, (n, c) = self.encoder_lstm(input_x,
                                                 (torch.zeros(1, self.batch_size, self.hidden_layer),
                                                  torch.zeros(1, self.batch_size, self.hidden_layer)))
        # decoder
        decoder_lstm, (n, c) = self.decoder_lstm(encoder_lstm,
                                                 (torch.zeros(1, self.batch_size, self.output_layer),
                                                  torch.zeros(1, self.batch_size, self.output_layer)))
        return decoder_lstm.squeeze()


class LstmFcAutoEncoder(nn.Module):
    def __init__(self, input_layer=300, hidden_layer=100, batch_size=20):
        super(LstmFcAutoEncoder, self).__init__()

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size

        self.encoder_lstm = nn.LSTM(self.input_layer, self.hidden_layer, batch_first=True)
        self.encoder_fc = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.decoder_lstm = nn.LSTM(self.hidden_layer, self.input_layer, batch_first=True)
        self.decoder_fc = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.relu = nn.ReLU()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        # encoder
        encoder_lstm, (n, c) = self.encoder_lstm(input_x,
                                                 # shape: (n_layers, batch, hidden_size)
                                                 (torch.zeros(1, self.batch_size, self.hidden_layer),
                                                  torch.zeros(1, self.batch_size, self.hidden_layer)))
        encoder_fc = self.encoder_fc(encoder_lstm)
        encoder_out = self.relu(encoder_fc)
        # decoder
        decoder_fc = self.relu(self.decoder_fc(encoder_out))
        decoder_lstm, (n, c) = self.decoder_lstm(decoder_fc,
                                                 (torch.zeros(1, 20, self.input_layer),
                                                  torch.zeros(1, 20, self.input_layer)))
        return decoder_lstm.squeeze()

def train(data_pth, resource, batch_size, epochs, save_pth):
    # 得到数据
    x, y = get_train_data(data_pth, resource)
    test_data = x[:20]
    y_real = y[:20]
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
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
            optimizer.zero_grad()
            # embed()
            y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            # y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
            # print("Train Step:", i, " loss: ", single_loss)
        # 每20次，输出一次前20个的结果，对比一下效果
        if (i + 1) % 5 == 0:
            # embed()
            y_pred = model(test_data).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            # print("Train epoch: ", i)
            # print("TEST: ", test_data)
            # print("PRED: ", y_pred)
            # print("LOSS: ", loss_function(y_pred, y_real))
            # from IPython import embed
            # embed()
            # test_data_ = test_data.detach().numpy()
            y_pred_ = y_pred.detach().numpy()

            print('MSE Value= ', mean_squared_error(y_pred_, y_real))
            print('MAE Value= ', mean_absolute_error(y_pred_, y_real))
            print('RMSE Value= ', sqrt(mean_squared_error(y_pred_, y_real)))
            print('MAPE Value= ', mean_absolute_percentage_error(y_pred_, y_real))

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

def test(data_pth, resource, test_len, model_pth, batch_size):
    # 定义损失参数
    total_MSE = 0
    total_MAE = 0
    total_RMSE = 0
    total_MAPE = 0

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
        y_pred = test_model(inputs).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
        y_pred_ = y_pred.detach().numpy()
        targets_ = targets.detach().numpy()

        MSE = mean_squared_error(y_pred_, targets_)
        MAE = mean_absolute_error(y_pred_, targets_)
        RMSE = sqrt(mean_squared_error(y_pred_, targets_))
        MAPE = mean_absolute_percentage_error(y_pred_, targets_)

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

    return test_MSE, test_MAE, test_RMSE, test_MAPE


if __name__ == '__main__':
    ### train
    # data_pth = '../data/machine_usage.csv'
    # resource = 'cpu'
    # batch_size = 20
    # epochs = 10
    # save_pth = '../output/' + 'auto_encoder_cpu_epoch_10.pth'
    #
    # train(data_pth, resource, batch_size, epochs, save_pth)

    ### test
    data_pth = '../data/machine_usage.csv'
    resource = 'cpu'
    test_len = 200
    model_pth = '../output/' + 'auto_encoder_cpu_epoch_10.pth'
    batch_size = 20

    MSE, MAE, RMSE, MAPE = test(data_pth, resource, test_len, model_pth, batch_size)
    print(MSE, MAE, RMSE, MAPE)
