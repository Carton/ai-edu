import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from torch.utils.data import TensorDataset, DataLoader
from HelperClass.NeuralNet_1_1 import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import warnings
warnings.filterwarnings('ignore')

file_name = "ch05.npz"


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    max_epoch = 500
    num_category = 3
    sdr = DataReader_1_1(file_name)
    sdr.ReadData()
    sdr.NormalizeX()
    sdr.NormalizeY()

    num_input = 2       # input size
    # get numpy form data
    XTrain, YTrain = sdr.XTrain, sdr.YTrain
    torch_dataset = TensorDataset(torch.FloatTensor(XTrain), torch.FloatTensor(YTrain))

    train_loader = DataLoader(          # data loader class
        dataset=torch_dataset,
        batch_size=32,
        shuffle=True,
    )

    loss_func = nn.MSELoss()  # 定义损失函数为均方误差损失
    model = Model(num_input)  # 创建模型实例
    # Adam（Adaptive Moment Estimation）优化器是一种自适应学习率的优化算法，它结合了Momentum和RMSprop两种优化方法的优点。
    optimizer = Adam(model.parameters(), lr=1e-4)  # 定义优化器为Adam，并设置学习率

    e_loss = []     # mean loss at every epoch
    for epoch in range(max_epoch):
        b_loss = []     # mean loss at every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()  # 清空梯度
            pred = model(batch_x)  # 预测
            loss = loss_func(pred, batch_y)  # 计算损失
            b_loss.append(loss.cpu().data.numpy())  # 将损失添加到b_loss列表中
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            #b_loss.append(loss.cpu().data.numpy())  # 将损失添加到b_loss列表中
        e_loss.append(np.mean(b_loss))  # 计算每个epoch的平均损失并添加到e_loss列表中
        if epoch % 20 == 0:
            print("Epoch: %d, Loss: %.5f" % (epoch, np.mean(b_loss)))  # 每20个epoch打印一次损失
    plt.plot([i for i in range(max_epoch)], e_loss)  # 绘制损失曲线
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Mean loss')  # y轴标签
    plt.show()  # 显示图像



