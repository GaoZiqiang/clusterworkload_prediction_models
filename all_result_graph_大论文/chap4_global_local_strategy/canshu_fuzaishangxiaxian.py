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

# 4.5.6 局部负载调度-负载上下限参数对应的系统吞吐量
def plot_total_strategy():
    X = [200,400,600,800,1000,1200,1400,1600,1800]
    Y1090 = [200,375,577,695,780,792,801,809,792]
    Y1080 = [200,398,599,714,805,818,821,812,803]
    Y2090 = [200,380,580,698,782,799,811,810,800]
    Y2080 = [200,400,598,720,800,815,823,822,804]
    Y1585 = [200,400,600,740,810,825,830,820,815]

    # AutoEncoder = [0.1113, 0.1482, 0.1587, 0.1554, 0.1727, 0.1745, 0.1876]
    # OurModel = [0.0751, 0.0666, 0.0766, 0.0980, 0.0988, 0.0983, 0.0998]

    plt.plot(X, Y1090, marker = 'h', color = 'blue', label = '10%-90%')
    plt.plot(X, Y1080, marker = '^', color = 'green', label = '10%-80%')
    plt.plot(X, Y2090, marker = 'o', color='black', label='20%-90%')
    plt.plot(X, Y2080, marker = 's', color='Cyan', label='20%-80%')
    plt.plot(X, Y1585, marker = 'p', color='red', label='15%-85%')# 叉号
    # plt.title('disk util percent prediction')


    plt.xlabel('用户并发请求数')# 并发请求数
    plt.ylabel('系统吞吐量')# 吞吐量
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_sequence()
    plot_total_strategy()
