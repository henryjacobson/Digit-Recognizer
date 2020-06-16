import importlib
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas

WIDTH = 28

class Data:
    def __init__(self, values, probs, labels):
        self.values = values
        self.probs = probs
        self.labels = labels

def csv_to_matrix(path):
    data = pandas.read_csv(path, delimiter = ',')
    mat = data.to_numpy()
    mat = mat.astype(np.float32)
    values = mat[:, 1:]
    labels = mat[:, 0:1].astype(np.int64)
    probs = np.zeros((labels.shape[0], 10))
    probs[np.arange(labels.shape[0]), labels.T] = 1
    values = values.reshape(data.shape[0], 1, WIDTH, WIDTH)
    values = torch.from_numpy(values)
    probs = probs.reshape(data.shape[0],10)
    probs = torch.from_numpy(probs)
    return Data(values, probs, labels)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 7, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(7),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(7, 7, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(7),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(7 * 7 * 7, 7 * 7 * 7),
            nn.Linear(7 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 7 * 7 * 7)
        x = self.linear_layers(x)
        return x

def evaluate(net, data):
    x = data.values
    out = net(x)
    out = out.detach().numpy()
    guess = np.apply_along_axis(np.argmax, 1, out)
    right = np.where(guess == data.labels.T[0], 1, 0)
    return sum(right) / data.labels.shape[0]

def train_net(net, n_epochs, learning_rate):
    data = csv_to_matrix('train.csv')
    half = data.values.shape[0] // 2
    x_train = data.values[0 : half]
    y_train = torch.from_numpy(data.labels[0 : half].T[0])
    x_val = data.values[half :]
    y_val = torch.from_numpy(data.labels[half :].T[0])

    net = net.float()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    #Use GPU if available
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()

    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        net.train()

        x_train_v, y_train_v = Variable(x_train), Variable(y_train)
        x_val_v, y_val_v = Variable(x_val), Variable(y_val)

        if torch.cuda.is_available():
            x_train_v, y_train_v = x_train_v.cuda(), y_train_v.cuda()
            x_val_v, y_val_v = x_val_v.cuda(), y_val_v.cuda()

        optimizer.zero_grad()

        output_train = net(x_train_v)
        output_val = net(x_val_v)

        loss_train = criterion(output_train, y_train_v)
        #train_losses.append(loss_train)
        loss_val = criterion(output_val, y_val_v)
        #val_losses.append(loss_val)

        loss_train.backward()
        optimizer.step()

        print('Finished epoch {}'.format(epoch))
    return train_losses, val_losses
