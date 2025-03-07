# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

class CLeakyRelu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        return np.array([x if x > 0 else self.alpha * x for x in z])

    def backward(self, z, a, delta):
        da = np.array([1 if x > 0 else self.alpha for x in a])
        #TODO: 原来写的是 dz = 0, 这里不能像下面这样写？
        dz = delta * da
        return da, dz
