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

# 4.5.4 局部动态负载调度算法折线图	不同负载时各节点任务分配情况
def plot_local_strategy():
    # X = [1,2,3,4]
    X = np.arange(1, 5, 1).astype(dtype=np.str)# 横坐标刻度只显示整数
    jiaoqing = [21,24,83,42]
    zhengchang = [38,40,147,95]
    jiaozhong = [45,53,192,120]

    # AutoEncoder = [0.1113, 0.1482, 0.1587, 0.1554, 0.1727, 0.1745, 0.1876]
    # OurModel = [0.0751, 0.0666, 0.0766, 0.0980, 0.0988, 0.0983, 0.0998]

    plt.plot(X, jiaoqing, marker = 's', color = 'red', label = '负载较轻')
    plt.plot(X, zhengchang, marker = 's', color = 'green', label = '负载正常')
    plt.plot(X, jiaozhong, marker = 's', color='purple', label='负载较重')

    # 设置数字标签
    for a, b in zip(X, jiaoqing):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(X, zhengchang):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(X, jiaozhong):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    plt.xlabel('服务器节点号')# 节点号
    plt.ylabel('运行任务数')# 任务数
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_sequence()
    plot_local_strategy()
