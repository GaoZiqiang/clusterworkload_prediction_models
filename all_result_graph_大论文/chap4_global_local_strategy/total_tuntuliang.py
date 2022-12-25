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

# 4.5.4 综合负载-系统吞吐量曲线图	
def plot_total_strategy():
    X = [200,400,600,800,1000,1200,1400,1600,1800]
    jianlun = [197,392,585,670,742,751,772,745,580]
    jialun = [198,395,590,680,750,760,780,755,600]
    zuilian = [199,396,592,685,755,764,800,774,631]
    DLBLF = [200,397,595,690,780,790,810,800,770]
    DLBDS = [200,398,600,700,790,800,810,805,782]
    OurModel = [200,400,600,740,810,820,830,820,810]

    # AutoEncoder = [0.1113, 0.1482, 0.1587, 0.1554, 0.1727, 0.1745, 0.1876]
    # OurModel = [0.0751, 0.0666, 0.0766, 0.0980, 0.0988, 0.0983, 0.0998]

    plt.plot(X, jianlun, linestyle='--', marker = 'h', color = 'blue', label = '简单轮询法')
    plt.plot(X, jialun, linestyle='--', marker = '^', color = 'green', label = '加权轮询法')
    plt.plot(X, zuilian, linestyle='--', marker = 'o', color='grey', label='最小连接数法')
    plt.plot(X, DLBLF, marker = 's', color='Cyan', label='DLBLF')
    plt.plot(X, DLBDS, marker = 'p', color='black', label='DLBDS')# 叉号
    plt.plot(X, OurModel, marker = 'd', color='red', label='本文模型')# diamond
    # plt.title('disk util percent prediction')


    plt.xlabel('用户并发请求数')# 用户并发请求数
    plt.ylabel('系统吞吐量')# 系统吞吐量
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_sequence()
    plot_total_strategy()
