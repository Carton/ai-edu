# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 1.1
- add MSE
- add Crossentropy for multiple classifier
"""

import numpy as np

from HelperClass.EnumDef_1_0 import *

class LossFunction_1_1(object):
    def __init__(self, net_type):
        self.net_type = net_type
    # end def

    # fcFunc: feed forward calculation
    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == 1:
            loss = self.MSE(A, Y, m)
        elif self.net_type == 2:
            loss = self.CE2(A, Y, m)
        elif self.net_type == 3:
            loss = self.CE3(A, Y, m)
        #end if
        return loss
    # end def

    def MSE(self, A, Y, count):
        p1 = A - Y #计算A-Y
        LOSS = np.multiply(p1, p1) #计算(A-Y)^2
        loss = LOSS.sum()/count/2 #计算平均损失
        return loss #返回平均损失
    # end def

    # for binary classifier
    def CE2(self, A, Y, count):
        p1 = 1 - Y #计算1-Y
        p2 = np.log(1 - A) #计算log(1-A)
        p3 = np.log(A) #计算log(A)

        p4 = np.multiply(p1 ,p2) #计算(1-Y)*log(1-A)
        p5 = np.multiply(Y, p3) #计算Y*log(A)

        LOSS = np.sum(-(p4 + p5))  #计算交叉熵损失
        loss = LOSS / count #计算平均损失
        return loss #返回平均损失
    # end def

    # for multiple classifier
    def CE3(self, A, Y, count):
        p1 = np.log(A) #计算log(A)
        p2 =  np.multiply(Y, p1) #计算Y*log(A)
        LOSS = np.sum(-p2) #计算交叉熵损失
        loss = LOSS / count #计算平均损失
        return loss #返回平均损失
    # end def
