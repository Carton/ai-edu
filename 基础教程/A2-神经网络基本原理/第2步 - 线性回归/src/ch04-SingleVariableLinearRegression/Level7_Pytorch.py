import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time  # Measure execution time
from torch.utils.data import TensorDataset, DataLoader
from HelperClass.DataReader_1_0 import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import warnings
warnings.filterwarnings('ignore')

file_name = "ch04.npz"


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # Check if GPU is available, and use it if possible
    start_time = time.time()  # Start time
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)

    max_epoch = 500
    num_category = 3
    sdr = DataReader_1_0(file_name)
    sdr.ReadData()

    num_input = 1       # input size
    # get numpy form data
    XTrain, YTrain = sdr.XTrain, sdr.YTrain
    torch_dataset = TensorDataset(torch.FloatTensor(XTrain).to(device), torch.FloatTensor(YTrain).to(device))

    batch_size = 64
    train_loader = DataLoader(          # data loader class
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loss_func = nn.MSELoss()
    # Initialize the model and move it to the device
    model = Model(num_input)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-2)

    e_loss = []     # mean loss at every epoch
    for epoch in range(max_epoch):
        b_loss = []     # mean loss at every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_func(pred,batch_y)
            if step == 0 and epoch % 20 == 0:
                print(f"step {step}:")
                # print(f"pred: {pred}")
                # print(f"batch_y: {batch_y}")
                print(f"loss: {loss}")
            b_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        e_loss.append(np.mean(b_loss))
        if epoch % 20 == 0:
            print("Epoch: %d, Loss: %.5f" % (epoch, np.mean(b_loss)))

    end_time = time.time()  # End time
    print("Execution time: {:.2f} seconds".format(end_time - start_time))  # Print execution time

    plt.plot([i for i in range(max_epoch)], e_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Mean loss')
    plt.show()



