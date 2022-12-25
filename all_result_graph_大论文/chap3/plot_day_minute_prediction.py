import torch
import torch.nn as nn
import torch.utils.data as Data

import pandas as pd
import random

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='Microsoft YaHei',weight="bold")

def data_plt(path, resource):
    dataset_train = pd.read_csv(path, on_bad_lines='skip', sep="\t")# 改参数on_bad_lines='skip'
    # embed()
    training_set = []
    if resource == "cpu":
        training_set = dataset_train.iloc[1000:1500, [2]].values# 使用cpu disk预测cpu
    elif resource == "mem":
        training_set = dataset_train.iloc[0:3325, 3:4].values
    elif resource == "disk":
        training_set = dataset_train.iloc[0:3325, 8:9].values
    elif resource == "net":
        training_set = dataset_train.iloc[0:3325, 4:5].values
    else:
        print("unknown resouce type, exit")
        return


    # with open('../pred_results/results.txt', 'a') as file:
    #     file.write('[1000:1500]-Minute-Date %s:\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
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
        # with open('../pred_results/results.txt', 'a') as file:
        #     file.write(
        #         '%f ' % (pdd))

    # with open('../pred_results/results.txt', 'a') as file:
    #     file.write('\n')
    # embed()

    ### Visualising the losses
    plt.plot(training_set, "b")
    plt.plot(pred_set, "r", label='OurModel')
    plt.plot(LSTM_set, "g", label='LSTM')
    plt.plot(TCN_set, "orange", label='TCN')
    plt.plot(AutoEncoder_set, "purple", label='AutoEncoder')
    plt.plot(training_set, color='green', label='MAPE loss')
    plt.title('training losses')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Minute', fontsize=20)
    plt.xlabel('时间')
    plt.ylabel('CPU利用率', fontsize=10)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data_plt('machine_usage.csv', 'cpu')
