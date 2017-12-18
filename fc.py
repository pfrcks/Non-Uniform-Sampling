from __future__ import print_function
import argparse
import torch
import sys
import struct
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
from multiprocessing.dummy import  Pool as ThreadPool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from logger import Logger

# Training settings
parser = argparse.ArgumentParser(description='Fully Connected Network')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='Which dataset to use: mnist or cifar')
parser.add_argument('--sample-type', type=str, default='grad', metavar='T',
                    help='Which sampling type to use: grad or obj or var or lev')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

logger = Logger('./logs')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

F_mnist, H_mnist, W_mnist = [1, 28, 28]
F_cifar, H_cifar, W_cifar = [3, 32, 32]
step = 0
reshape_size = 0
if args.dataset == 'mnist':
    reshape_size = F_mnist * H_mnist * W_mnist
else:
    reshape_size = F_cifar * H_cifar * W_cifar

def read_mnist(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def read_cifar(path):

    # training data
    data = [np.load(os.path.join(path, 'cifar-10-batches-py',
                                 'data_batch_%d' % (i + 1))) for i in range(5)]
    x_train = np.vstack([d['data'] for d in data])
    y_train = np.hstack([np.asarray(d['labels'], np.int32) for d in data])

    # test data
    data = np.load(os.path.join(path, 'cifar-10-batches-py', 'test_batch'))
    x_test = data['data']
    y_test = np.asarray(data['labels'], np.int32)

    return x_train, y_train, x_test, y_test

def load_dataset(type, path='../data'):
    x_train = x_test = y_train = y_test = None
    if type == 'mnist':
        Ntr, F, H, W = 60000, 1, 28, 28
        Nte = 10000

        x_train = read_mnist('../data/raw/train-images-idx3-ubyte')
        x_test = read_mnist('../data/raw/t10k-images-idx3-ubyte')
        y_train = read_mnist('../data/raw/train-labels-idx1-ubyte')
        y_test = read_mnist('../data/raw/t10k-labels-idx1-ubyte')

        x_train = x_train.reshape(Ntr, F, H, W)
        x_test = x_test.reshape(Nte, F, H, W)

    elif type == 'cifar':
        F, H, W = 3, 32, 32
        x_train, y_train, x_test, y_test = read_cifar(path)

        # reshape
        x_train = x_train.reshape(-1, F, H, W)
        x_test = x_test.reshape(-1, F, H, W)

        # normalize
        try:
            mean_std = np.load(os.path.join(path, 'cifar-10-mean_std.npz'))
            mean = mean_std['mean']
            std = mean_std['std']
        except IOError:
            mean = x_train.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
            std = x_train.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
            np.savez(os.path.join(path, 'cifar-10-mean_std.npz'),
                     mean=mean, std=std)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    return x_train, y_train, x_test, y_test

def to_np(x):
    out = x.data
    if args.cuda:
        out = x.data.cpu()
    return out.numpy()

def to_var(x, volatile=False):
    if args.cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)

def torch_data(x, y):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)

    return x, y


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


class CIFAR_Net(nn.Module):
    def __init__(self):
        super(CIFAR_Net, self).__init__()
        self.fc1 = nn.Linear(3072, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

def train(x, y, ret_all_losses=False):
    model.train()
    x, y = torch_data(x, y)
    data, target = to_var(x), to_var(y)
    optimizer.zero_grad()
    output = model(data)

    all_losses = None
    if ret_all_losses:
        all_losses = F.nll_loss(output, target, reduce=False)
        if args.cuda:
            all_losses = all_losses.cpu().data.numpy()
        else:
            all_losses = all_losses.data.numpy()

    loss = F.nll_loss(output, target)
    val_loss = loss.data[0]
    if idx % 10 == 0:
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        print('\n train set loss: ', val_loss/x.shape[0], correct/float(x.shape[0]))

    loss.backward()
    optimizer.step()
    return val_loss, all_losses

def test(x, y):
    model.eval()
    test_loss = 0
    correct = 0

    x, y = torch_data(x, y)
    data, target = to_var(x, volatile=True), to_var(y, volatile=True)

    output = model(data)
    test_loss += F.nll_loss(output, target, size_average=False).data[0]
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= x_test.shape[0]
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, x_test.shape[0],
        100. * correct / x_test.shape[0]))

def bernoulli_sample(score, sample_size):
    sel = np.random.binomial(1, prob)
    return sel


def gradients(x, y):
    n, f, h, w = x.shape
    x, y = torch_data(x, y)
    data, target = to_var(x), to_var(y)

    weight_grads = np.zeros((n,))
    for i in np.arange(n):
        output = model(data[i].view(-1, f, h, w))
        loss = F.nll_loss(output, target[i])
        loss.backward()

        model_params = list(model.parameters())
        norm = 0
        for param in model_params:
            grad = to_np(param.grad)
            norm = np.sqrt((grad * grad).sum() + (norm * norm))
        weight_grads[i] = norm
        optimizer.zero_grad()

    return weight_grads

def uniform_sampling(x, y):
    n = x.shape[0]
    size = args.batch_size
    uniform_epoch = []
    p = np.random.permutation(n)
    x = x[p]
    y = y[p]
    for i in range(n/size):
        global step
        step += 1
        loss, _ = train(x[i*size: (i+1)*size], y[i*size: (i+1)*size])
        uniform_epoch.append(loss)
    return uniform_epoch

def gradient_sampling(x, y, prob):
    n = x.shape[0]
    size = args.batch_size
    gradient_epoch = []
    for i in range(n/size):
        global step
        step += 1
        sel = np.random.binomial(1, prob)
        x_batch = x[np.where(sel == 1)]
        y_batch = y[np.where(sel == 1)]
        loss, _ = train(x_batch, y_batch)
        gradient_epoch.append(loss)
    return gradient_epoch

def variance_sampling(x, y, prob):
    n = x.shape[0]
    size = args.batch_size
    variance_epoch = []
    for i in range(n/size):
        global step
        step += 1
        sel = np.random.binomial(1, prob)
        x_batch = x[np.where(sel == 1)]
        y_batch = y[np.where(sel == 1)]
        loss, _ = train(x_batch, y_batch)
        variance_epoch.append(loss)
    return variance_epoch

def objective_sampling(x, y, losses):
    n = x.shape[0]
    size = args.batch_size
    objective_epoch = []
    for i in range(n/size):
        global step
        step += 1
        ind = np.argsort(losses[:, 0])
        losses = losses[ind][::-1]
        idx = losses[:, 1][:size]
        idx = idx.astype(int)
        loss, loss_batch = train(x[idx], y[idx], ret_all_losses=True)
        losses[:, 0][:size] = loss_batch
        objective_epoch.append(loss)
    return objective_epoch, losses


if args.dataset == 'mnist':
    model = MNIST_Net()
elif args.dataset == 'cifar':
    model = CIFAR_Net()

if args.cuda:
    model.cuda()

x_train, y_train, x_test, y_test = load_dataset(args.dataset)
n = x_train.shape[0]
x_train = x_train[:n]
y_train = y_train[:n]

optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.sample_type == 'grad':
    for epoch in np.arange(3):
        uniform_sampling(x_train, y_train)
        test(x_test, y_test)
    weights = gradients(x_train, y_train)
    weights = weights.reshape(weights.shape[0], 1)
    weights = np.column_stack([weights, range(weights.shape[0])])
    prob = args.batch_size * (weights[:, 0] / np.sum(weights[:, 0]))
    for epoch in np.arange(args.epochs-3):
        gradient_sampling(x_train, y_train, prob)
        test(x_test, y_test)
elif args.sample_type == 'obj':
    loss, losses = train(x_train, y_train, ret_all_losses=True)
    losses = losses.reshape(losses.shape[0], 1)
    losses = np.column_stack([losses, range(losses.shape[0])])
    for epoch in np.arange(args.epochs-1):
        objective_epoch, losses = objective_sampling(x_train, y_train, losses)
        test(x_test, y_test)
elif args.sample_type == 'var':
    x_train_var = np.var(x_train.reshape(x_train.shape[0], reshape_size), axis=1)
    prob = args.batch_size * (x_train_var / np.sum(x_train_var))
    for epoch in np.arange(args.epochs):
        variance_sampling(x_train, y_train, prob)
        test(x_test, y_test)
elif args.sample_type == 'uni':
    for epoch in np.arange(args.epochs):
        uniform_sampling(x_train, y_train)
        test(x_test, y_test)
