import numpy as np
import matplotlib.pyplot as plt


def plot_labels():
    x = [1,2,3,4,5]
    y = [0.1,0.2,3,4,5]

    plt.plot(x,y,'o')

    for x, y in zip(x, y):
        # print("x=",x)
        if x==1 or x==5:
            plt.text(x, y, '%.2f' % y, fontdict={'fontsize': 14})

    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_labels()