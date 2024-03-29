import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.rcParams['font.sans-serif'] = ['SimHei'] #或者把"SimHei"换为"KaiTi"
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

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

def plot_sequence():
    X = [1,10,30,60,90,120,150]
    MAE = [0.0513,0.0342,0.0286,0.0491,0.0446,0.0484,0.0458]
    RMSE = [0.0858,0.0750,0.0421,0.0650,0.0685,0.0670,0.0634]
    MAPE = [0.0751,0.0666,0.0766,0.0980,0.0988,0.0983,0.0998]

    # AutoEncoder = [0.1113, 0.1482, 0.1587, 0.1554, 0.1727, 0.1745, 0.1876]
    # OurModel = [0.0751, 0.0666, 0.0766, 0.0980, 0.0988, 0.0983, 0.0998]

    plt.plot(X, MAE, marker = 'h', color = 'green', label = 'MAE')
    plt.plot(X, RMSE, marker = '^', color = 'blue', label = 'RMSE')
    plt.plot(X, MAPE, marker = 'o', color='red', label='MAPE')
    # plt.title('disk util percent prediction')

    for x, y1, y2, y3 in zip(X, MAE,RMSE,MAPE):
        # print("x=",x)
        if x==1 or x==150:
            plt.text(x, y1, '%.4f' % y1, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            plt.text(x, y2, '%.4f' % y2, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            plt.text(x, y3, '%.4f' % y3, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            #plt.text(x, y4, '%.4f' % y4, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")

    plt.xlabel('滑动窗口长度')
    plt.ylabel('损失值')
    plt.legend(prop = {'size':9})# 设置图例大小
    plt.show()

def plot_step():
    X = [1,2,3,4,5,6,7,8,9,10,11,12]

    LSTM = [0.1100, 0.1197, 0.1191,0.1185, 0.1191,0.1197, 0.1208,0.1219, 0.1230,0.1241, 0.1264,0.1286]
    TCN = [0.1700, 0.1742, 0.1830,0.1917, 0.2065,0.2213, 0.2317,0.2421, 0.2502,0.2582, 0.2569,0.2555]
    AutoEncoder = [0.0900, 0.0989, 0.1069,0.1148, 0.1143,0.1137, 0.1172,0.1207, 0.1243,0.1279, 0.1329,0.1378]
    OurModel = [0.0400, 0.0490, 0.0741,0.0992, 0.1035,0.1077, 0.1052,0.1027, 0.1064,0.1101, 0.1133,0.1164]


    # LSTM = [0.1240, 0.1185, 0.1197, 0.1219, 0.1241, 0.1286]
    # TCN = [0.1742, 0.1917, 0.2213, 0.2421, 0.2582, 0.2555]
    # AutoEncoder = [0.0989, 0.1148, 0.1137, 0.1207, 0.1279, 0.1378]
    # OurModel = [0.0490, 0.0992, 0.1077, 0.1027, 0.1101, 0.1164]

    # AutoEncoder = [0.1113, 0.1482, 0.1587, 0.1554, 0.1727, 0.1745, 0.1876]
    # OurModel = [0.0751, 0.0666, 0.0766, 0.0980, 0.0988, 0.0983, 0.0998]

    plt.plot(X, LSTM, marker = 'p', color = 'green', label = 'LSTM')
    plt.plot(X, TCN, marker = '^', color = 'blue', label = 'TCN')
    plt.plot(X, AutoEncoder, marker = 'o', color='red', label='AutoEncoder')
    plt.plot(X, OurModel, marker = 's', color='grey', label='OurModel')
    # plt.title('disk util percent prediction')

    for x, y1, y2, y3, y4 in zip(X, LSTM,TCN,AutoEncoder,OurModel):
        # print("x=",x)
        if x==1 or x==12:
            plt.text(x, y1, '%.4f' % y1, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            plt.text(x, y2, '%.4f' % y2, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            plt.text(x, y3, '%.4f' % y3, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            plt.text(x, y4, '%.4f' % y4, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")

    plt.xlabel('prediction step')
    plt.ylabel('MAPE')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_sequence()
    plot_sequence()
