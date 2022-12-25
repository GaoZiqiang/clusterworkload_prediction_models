import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.font_manager import FontProperties

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

# 4.5.4 各模型的负载均衡度实验结果与折线图
def plot_global_local_strategy():
    X = [0,10,20,30,40,50,60]
    jianlun = [0,0,2.4,7.1,6.6,5.0,4.8]
    jialun = [0,0,2.0,6.15,4.78,4.7,4.2]
    zuilian = [0,0,1.6,6.2,4.5,4.9,4.85]
    DLBLF = [0,0,1.3,5.1,5.6,3.1,3.7]
    DLBDS = [0,0,1.60,5.55,6.0,3.7,4.0]
    OurModel = [0,0,0.2,2.5,2.3,2.0,2.1]

    # AutoEncoder = [0.1113, 0.1482, 0.1587, 0.1554, 0.1727, 0.1745, 0.1876]
    # OurModel = [0.0751, 0.0666, 0.0766, 0.0980, 0.0988, 0.0983, 0.0998]
    # 在此设置字体及大小
    # font = FontProperties(fname=r"/root/whq/font/SimSun.ttf", size=14)

    plt.plot(X, jianlun, linestyle='--', marker = 'h', color = 'blue', label = '简单轮询法')
    plt.plot(X, jialun, linestyle='--', marker = '^', color = 'green', label = '加权轮询法')
    plt.plot(X, zuilian, linestyle='--', marker = 'o', color='grey', label='最小连接数法')
    plt.plot(X, DLBLF, marker = 's', color='Cyan', label='DLBLF')
    plt.plot(X, DLBDS, marker = 'p', color='black', label='DLBDS')# 叉号
    plt.plot(X, OurModel, marker = 'd', color='red', label='本文模型')# diamond
    # plt.title('disk util percent prediction')


    plt.xlabel("用户请求时长(s)")# 时间
    plt.ylabel("负载均衡度")# 负载均衡度
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_sequence()
    plot_global_local_strategy()
