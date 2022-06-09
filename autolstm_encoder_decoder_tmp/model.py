import torch
import torch.nn as nn
import torch.utils.data as Data

import numpy as np

from utils import *

from IPython import embed

class LstmAutoEncoder(nn.Module):
    # 测试不同window size之前 该模型是没有window_size这个参数的
    def __init__(self, num_features=2, hidden_size=16, hidden_layers=2, output_features=1, batch_size=20):
        super(LstmAutoEncoder, self).__init__()

        self.num_features = num_features# 输入特征数 比如只使用cpu和mem两个特征
        self.hidden_size = hidden_size# 隐藏层size
        self.output_features = output_features# 输出特征数/预测特征数
        self.hidden_layers = hidden_layers# 隐藏层的个数
        # self.window_size = window_size# 滑动窗口长度
        self.num_directions = 1# 单向LSTM
        self.batch_size = batch_size
        # 添加一层CNN
        self.conv1d = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=3, stride=1, padding=1)
        self.encoder_lstm = nn.LSTM(self.num_features, self.hidden_size, self.hidden_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(self.hidden_size, self.output_features, self.hidden_layers, batch_first=True)

    def forward(self, input_x):
        # 1D CNN
        input_x = self.conv1d(input_x)

        # batch_size sliding_window features_num
        # input_x = input_x.view(20, 120, 2)
        input_x = input_x.view(self.batch_size, 120, self.num_features)# batch_size slidingwindow feature数

        # encoder
        # 输入形参
        # self.num_directions * self.num_layers self.batch_size self.hidden_size
        encoder_lstm, (n, c) = self.encoder_lstm(input_x,
                                                 (torch.zeros(self.num_directions*self.hidden_layers, self.batch_size, self.hidden_size),
                                                  torch.zeros(self.num_directions*self.hidden_layers, self.batch_size, self.hidden_size)))
        # decoder
        decoder_lstm, (n, c) = self.decoder_lstm(encoder_lstm,
                                                 (torch.zeros(self.num_directions * self.hidden_layers, self.batch_size, self.output_features),
                                                  torch.zeros(self.num_directions * self.hidden_layers, self.batch_size, self.output_features)))
        # embed()
        # print("decoder_lstm.shape = ", decoder_lstm.shape)
        return decoder_lstm

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

def Conv1D():
    # padding的设置(下面的计算不靠谱) --> 暂时将padding设置为1
    # padding = (kernel_size-1)*dilation dilation默认为1
    # model_m = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1))
    model_m = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    # model_m.add(Conv1D(in_channels=2, out_channels=2, kernel_size=3, padding='same'))
    print(model_m)

    return model_m

if __name__ == "__main__":
    data_pth = '../data/machine_usage.csv'
    resource = 'cpu'
    x, y = get_train_data(data_pth, resource)
    # train_loader = Data.DataLoader(
    #     dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
    #     batch_size=20,  # 每块的大小
    #     shuffle=False,  # 要不要打乱数据 (打乱比较好)
    #     num_workers=2,  # 多进程（multiprocess）来读数据
    # )
    # model = LstmAutoEncoder()  # lstm
    # loss_function = nn.MSELoss()  # loss
    # for input, target in train_loader:
    #     embed()
    #     y_pred = model(input)
    #     y_pred = y_pred[:, -1, :]
    #     loss = loss_function(y_pred.squeeze(), target)
    model = Conv1D()
    result = model(x)
    from IPython import embed
    embed()
    # print(result)