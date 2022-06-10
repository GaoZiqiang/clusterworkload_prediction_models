import matplotlib.pyplot as plt
import numpy as np

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
    X = [1,15,30,45,60,75,90,105,120,135,150]
    LSTM = [0.0848,0.1504,0.1624,0.1603,0.1582,0.1577,0.1571,0.1659,0.1746,0.1756,0.1765]
    TCN = [0.1056,0.1410,0.1483,0.1485,0.1488,0.1504,0.1519,0.1528,0.1536,0.1561,0.1586]
    AutoEncoder = [0.1113,0.1535,0.1587,0.1571,0.1554,0.1641,0.1727,0.1736,0.1745,0.1811,0.1876]
    OurModel = [0.0751,0.0716,0.0766,0.0873,0.0980,0.0984,0.0988,0.0986,0.0983,0.0991,0.0998]

    # AutoEncoder = [0.1113, 0.1482, 0.1587, 0.1554, 0.1727, 0.1745, 0.1876]
    # OurModel = [0.0751, 0.0666, 0.0766, 0.0980, 0.0988, 0.0983, 0.0998]

    plt.plot(X, LSTM, marker = 'h', color = 'green', label = 'LSTM')
    plt.plot(X, TCN, marker = '^', color = 'blue', label = 'TCN')
    plt.plot(X, AutoEncoder, marker = 'o', color='red', label='AutoEncoder')
    plt.plot(X, OurModel, marker = 's', color='grey', label='OurModel')
    # plt.title('disk util percent prediction')

    for x, y1, y2, y3, y4 in zip(X, LSTM,TCN,AutoEncoder,OurModel):
        # print("x=",x)
        if x==1 or x==150:
            plt.text(x, y1, '%.4f' % y1, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            plt.text(x, y2, '%.4f' % y2, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            plt.text(x, y3, '%.4f' % y3, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")
            plt.text(x, y4, '%.4f' % y4, fontdict={'fontsize': 10},verticalalignment="bottom",horizontalalignment="center")

    plt.xlabel('prediction window size')
    plt.ylabel('MAPE')
    plt.legend()
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
    plot_step()