import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rc("font",family='Microsoft YaHei',weight="bold")

def plot():
    labels = ['1:2', '1:1.5', '1:1', '1.5:1', '2:1']  # 级别
    men_means = [5.215, 5.133, 4.625, 4.114, 4.790]  # 平均时延
    women_means = [0.210, 0.217, 0.225, 0.216, 0.224]  # 最短时延
    wm_means = [16.010, 15.491, 14.150, 12.126, 13.835]  # 最长时延

    x = np.arange(len(men_means))


    plt.figure(figsize=(27, 12))

    width = 0.95

    rects1 = plt.bar(x - width / 3, men_means, width / 3)  # 返回绘图区域对象
    rects2 = plt.bar(x, women_means, width / 3)
    rects3 = plt.bar(x + width / 3, wm_means, width / 3)

    # 设置标签标题，图例
    plt.ylabel('时间', fontsize=25)
    plt.yticks(fontsize=22)
    # plt.title('Scores by group and gender')
    plt.xticks(x, labels, rotation=0, fontsize=30)
    plt.legend(['pingjun', 'zuiduan', 'zuichang'], fontsize=20)

    # 添加注释
    def set_label(rects):
        for rect in rects:
            height = rect.get_height()  # 获取⾼度
            plt.text(x=rect.get_x() + rect.get_width() / 2,  # ⽔平坐标
                     y=height + 0.005,  # 竖直坐标
                     s=height,  # ⽂本
                     ha='center', fontsize=0)  # ⽔平居中

    set_label(rects1)
    set_label(rects2)
    set_label(rects3)

    print("end")
if __name__ == "__main__":
    plot()
# plt.tight_layout() # 设置紧凑布局
# plt.savefig('./分组带标签柱状图.png')