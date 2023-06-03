# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass.NeuralNet_1_2 import *

file_name = "ch09_1.train.npz"

class DataReaderEx(DataReader_1_3):
    def Add(self):
        X = self.XTrain[:,]**2
        self.XTrain = np.hstack((self.XTrain, X)) # 把一个特征的平方作为新的特征
        X = self.XTrain[:,0:1]**3
        self.XTrain = np.hstack((self.XTrain, X)) # 把一个特征的立方作为新的特征
        # X = self.XTrain[:,0:1]**4
        # self.XTrain = np.hstack((self.XTrain, X)) # 把一个特征的4次方作为新的特征


def ShowResult(net, dataReader, title):
    # draw train data
    X,Y = dataReader.XTrain, dataReader.YTrain
    plt.plot(X[:,0], Y[:,0], '.', c='b')
    # create and draw visualized validation data
    TX1 = np.linspace(0,1,100).reshape(100,1)
    TX = np.hstack((TX1, TX1[:,]**2))
    TX = np.hstack((TX, TX1[:,]**3))
    # TX = np.hstack((TX, TX1[:,]**4))

    TY = net.inference(TX)
    plt.plot(TX1, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def

if __name__ == '__main__':
    dataReader = DataReaderEx(file_name)
    dataReader.ReadData()
    dataReader.Add()
    print(dataReader.XTrain.shape)

    # 这里是把输入的 x 转换为 x^2, x^3, x^4，然后用 4 个特征来拟合
    # 修改为只用 3 个特征：x, x^2, x^3，发现也可以拟合
    # net
    num_input = 3
    num_output = 1
    params = HyperParameters_1_1(num_input, num_output, eta=0.2, max_epoch=10000, batch_size=10, eps=0.005, net_type=NetType.Fitting)
    net = NeuralNet_1_2(params)
    net.train(dataReader, checkpoint=10)
    ShowResult(net, dataReader, params.toString())
