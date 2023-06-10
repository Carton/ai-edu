# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HelperClass2.NeuralNet_2_0 import *
from HelperClass2.DataReader_2_0 import *

train_data_name = "ch09_1.train.npz"
test_data_name = "ch09_1.test.npz"

def ShowResult(net, dataReader, title):
    # draw train data
    X,Y = dataReader.XTrain, dataReader.YTrain
    plt.plot(X[:,0], Y[:,0], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def

if __name__ == '__main__':
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    n_input, n_hidden, n_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.05, 10, 30000
    eps = 0.002

    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet_2_0(hp, "sin_121")
    #net.LoadResult()  # 把之前训练的结果加载进来，继续训练

    net.train(dataReader, 50, True)
    net.ShowTrainingHistory()
    ShowResult(net, dataReader, hp.toString())
