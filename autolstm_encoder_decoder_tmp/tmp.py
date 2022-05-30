import matplotlib.pyplot as plt

if __name__ == "__main__":
    total_MSE = [1,2]
    total_MAPE = [10,20]

    plt.plot(total_MSE, color='red', label='MSE loss')
    plt.plot(total_MAPE, color='green', label='MAPE loss')
    plt.title('training losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    epoch = [x for x in range(len(total_MSE))]
    # print(epoch)
    for a, b in zip(epoch, total_MSE):
        plt.text(a, b, b, ha="center", va="bottom", fontsize=10)

    for a, b in zip(epoch, total_MAPE):
        plt.text(a, b, b, ha="center", va="bottom", fontsize=10)

    plt.legend()
    plt.show()
    # print("end")