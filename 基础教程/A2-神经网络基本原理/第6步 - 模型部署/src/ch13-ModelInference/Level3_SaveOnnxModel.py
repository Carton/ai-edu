import os
import numpy as np
from ONNXConverter.conv2d import Cconv2d
from ONNXConverter.pool import Cpool
from ONNXConverter.relu import Crelu
from ONNXConverter.softmax import Csoftmax
from ONNXConverter.sigmoid import Csigmoid
from ONNXConverter.tanh import Ctanh
from ONNXConverter.fc import Cfc
from ONNXConverter.dropout import Cdropout
from ONNXConverter.save import model_save
from ONNXConverter.transfer import ModelTransfer


# some configuration
# these variables records the path for the stored weights
fc1_wb_path = "./MNIST_64_16/wb1.npz"
fc2_wb_path = "./MNIST_64_16/wb2.npz"
fc3_wb_path = "./MNIST_64_16/wb3.npz"

fc1_wb = np.load(fc1_wb_path)  # 加载第一层全连接层的权重和偏置
fc2_wb = np.load(fc2_wb_path)  # 加载第二层全连接层的权重和偏置
fc3_wb = np.load(fc3_wb_path)  # 加载第三层全连接层的权重和偏置

outputshape1 = fc1_wb['weights'].shape[1]  # 第一层全连接层的输出形状
outputshape2 = fc2_wb['weights'].shape[1]  # 第二层全连接层的输出形状
outputshape3 = fc3_wb['weights'].shape[1]  # 第三层全连接层的输出形状
print(outputshape1, outputshape2, outputshape3)

# 定义模型结构
class Cmodel(object):
    def __init__(self):
        """模型定义."""
        self.fc1 = Cfc([1,784], outputshape1, name="fc1", exname="")  # 第一层全连接层
        self.activation1 = Csigmoid(self.fc1.outputShape, name="activation1", exname="fc1")  # 第一层激活函数
        self.fc2 = Cfc(self.fc1.outputShape, outputshape2, name="fc2", exname="activation1")  # 第二层全连接层
        self.activation2 = Ctanh(self.fc2.outputShape, name="activation2", exname="fc2")  # 第二层激活函数
        self.fc3 = Cfc(self.fc2.outputShape, outputshape3, name="fc3", exname="activation2")  # 第三层全连接层
        self.activation3 = Csoftmax(self.fc3.outputShape, name="activation3", exname="fc3")  # 第三层激活函数

        self.model = [
          self.fc1, self.activation1, self.fc2, self.activation2, self.fc3, self.activation3,
        ]

        self.fc1.weights = fc1_wb['weights']  # 加载第一层全连接层的权重
        self.fc1.bias = fc1_wb['bias'].reshape(self.fc1.bias.shape)  # 加载第一层全连接层的偏置
        self.fc2.weights = fc2_wb['weights']  # 加载第二层全连接层的权重
        self.fc2.bias = fc2_wb['bias'].reshape(self.fc2.bias.shape)  # 加载第二层全连接层的偏置
        self.fc3.weights = fc3_wb['weights']  # 加载第三层全连接层的权重
        self.fc3.bias = fc3_wb['bias'].reshape(self.fc3.bias.shape)  # 加载第三层全连接层的偏置

    def save_model(self, path="./"):
        model_path = os.path.join(path, "model.json")
        model_save(self.model, path)
        ModelTransfer(model_path, os.path.join(path, "mnist.onnx"))

if __name__ == '__main__':

    # attention: need ONNX 1.4.1 python package
    # pip install --upgrade onnx

    model = Cmodel()
    save_path = 'ONNX'
    model.save_model(save_path)
    print(f'Succeed! Your model file is {os.path.join(save_path, "mnist.onnx")}')


