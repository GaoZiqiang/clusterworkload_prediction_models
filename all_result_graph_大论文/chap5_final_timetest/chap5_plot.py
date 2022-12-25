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

# 5.1.1 系统与模型时间性能评估结果	
def plot_time():
    X = [0.1,0.5,3,6,12,24,48,72]
    xitong = [0.31,0.44,0.65,0.82,0.94,1.25,1.77,2.05]
    moxing = [0.23,0.36,0.43,0.52,0.71,1.07,1.62,1.84]

    # AutoEncoder = [0.1113, 0.1482, 0.1587, 0.1554, 0.1727, 0.1745, 0.1876]
    # OurModel = [0.0751, 0.0666, 0.0766, 0.0980, 0.0988, 0.0983, 0.0998]

    plt.plot(X, xitong, marker = 'h', color = 'red', label = '系统')
    plt.plot(X, moxing, marker = '^', color = 'blue', label = '模型')


    plt.xlabel('历史负载序列长度', fontsize = 15)# 历史负载序列长度
    plt.ylabel('耗时', fontsize = 15)# 时间
    plt.legend(fontsize = 15)
    plt.show()

if __name__ == "__main__":
    # plot_sequence()
    plot_time()
