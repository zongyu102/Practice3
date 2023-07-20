import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

def data_read(dir_path):#读取一维数组
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)

def draw(filepath, savename):
    y = data_read(filepath)
    x = range(len(y))
    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')
    plt.ylabel(savename)
    plt.plot(x, y, color='blue', linestyle='solid', label=savename)
    if savename == 'L_train_loss' or savename == 'G_train_loss':
        plt.title('Train Loss')
    elif savename == 'L_train_acc' or savename == 'G_train_acc':
        plt.title('Train Acc')
    elif savename == 'L_test_loss' or savename == 'G_test_loss':
        plt.title('Test Loss')
    elif savename == 'L_test_acc' or savename == 'G_test_acc':
        plt.title('Test Acc')


    path = './DataImage'
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = path + '/' + savename + '.png'
    plt.savefig(savepath)
    plt.show()

def draw2im(path1, path2, savename):
    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\MSYHL.TTC")
    y1 = data_read(path1)
    y2 = data_read(path2)
    x = range(len(y1))
    plt.figure()
    plt.plot(x, y1, label='LSTM', color='red')
    plt.plot(x, y2, label='GRU', color='blue')
    plt.xticks(x[::10])
    plt.xlabel('epoch')
    if savename == 'train_compare_loss':
        plt.ylabel('train loss')
    elif savename == 'train_compare_acc':
        plt.ylabel('train acc')
    elif savename == 'test_compare_loss':
        plt.ylabel('test loss')
    elif savename == 'test_compare_acc':
        plt.ylabel('test acc')
    plt.grid(alpha=0.6, linestyle='-')
    plt.legend(prop=my_font, loc='upper left')

    path = './DataImage'
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = path + '/' + savename + '.png'
    plt.savefig(savepath)
    plt.show()

if __name__ == "__main__":
    L_train_loss = './Data/LSTM_train_epoch_loss.txt'
    L_train_loss_name = 'L_train_loss'
    draw(L_train_loss, L_train_loss_name)

    L_train_acc = './Data/LSTM_train_acc.txt'
    L_train_acc_name = 'L_train_acc'
    draw(L_train_acc, L_train_acc_name)

    L_test_loss = './Data/LSTM_test_epoch_loss.txt'
    L_test_loss_name = 'L_test_loss'
    draw(L_test_loss, L_test_loss_name)

    L_test_acc = './Data/LSTM_test_acc.txt'
    L_test_acc_name = 'L_test_acc'
    draw(L_test_acc, L_test_acc_name)

    G_train_loss = './Data/GRU_train_epoch_loss.txt'
    G_train_loss_name = 'G_train_loss'
    draw(G_train_loss, G_train_loss_name)

    G_train_acc = './Data/GRU_train_acc.txt'
    G_train_acc_name = 'G_train_acc'
    draw(G_train_acc, G_train_acc_name)

    G_test_loss = './Data/GRU_test_epoch_loss.txt'
    G_test_loss_name = 'G_test_loss'
    draw(G_test_loss, G_test_loss_name)

    G_test_acc = './Data/GRU_test_acc.txt'
    G_test_acc_name = 'G_test_acc'
    draw(G_test_acc, G_test_acc_name)

    draw2im(L_test_loss, G_test_loss, 'test_compare_loss')
    draw2im(L_test_acc, G_test_acc, 'test_compare_acc')
