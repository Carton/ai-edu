# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.DataReader_1_1 import *

file_name = "ch05.npz"

if __name__ == '__main__':
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    print(f"X.shape: {X.shape}")
    for num_example in range(100, X.shape[0]+100, 100):
        print(f"num_example: {num_example}")
        one = np.ones((num_example,1))
        x = np.column_stack((one, (X[0:num_example,:])))
        #print(x)

        a = np.dot(x.T, x)
        # need to convert to matrix, because np.linalg.inv only works on matrix instead of array
        b = np.asmatrix(a)
        #print(b)
        c = np.linalg.inv(b)
        d = np.dot(c, x.T)
        e = np.dot(d, Y[:num_example])
        #print(e)
        b=e[0,0]
        w1=e[1,0]
        w2=e[2,0]
        print("w1=", w1)
        print("w2=", w2)
        print("b=", b)
        # inference
        z = w1 * 15 + w2 * 93 + b
        print("z=",z)
        print() 
