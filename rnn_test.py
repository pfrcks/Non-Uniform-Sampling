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
from sklearn.cross_validation import train_test_split

# Training settings
parser = argparse.ArgumentParser(description='RNN')
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
parser.add_argument('--cross-val', action='store_true', default=False,
                    help='Perform cross validation')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

logger = Logger('./logs')
np.random.seed(0)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

F_mnist, H_mnist, W_mnist = [1, 28, 28]
F_cifar, H_cifar, W_cifar = [3, 32, 32]
step = 0
reshape_size = 0
if args.dataset == 'mnist':
    reshape_size = F_mnist * H_mnist * W_mnist
elif args.dataset == 'cifar':
    reshape_size = F_cifar * H_cifar * W_cifar
elif args.dataset == 'sine':
    pass

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

def read_sine(path, test=1000):
    data = torch.load('traindata.pt')
    x_train = data[test:, :-1]
    y_train = data[test:, 1:]
    x_test = data[:test, :-1]
    y_test = data[:test, 1:]
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
    elif type == 'sine':
        T = 20
        L = 1000
        N = 500
        test = 100

        x = np.empty((N, L), 'int64')
        x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
        data = np.sin(x / 1.0 / T).astype('float64')
        x_train = data[test:, :-1]
        y_train = data[test:, 1:]
        x_test = data[:test, :-1]
        y_test = data[:test, 1:]
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test, x_val, y_val

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
    if args.dataset == 'sine':
        y = torch.from_numpy(y).type(torch.FloatTensor)
    else:
        y = torch.from_numpy(y).type(torch.LongTensor)

    return x, y


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class CIFAR_Net(nn.Module):
    def __init__(self):
        super(CIFAR_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        if args.cuda:
            h_t = Variable(torch.zeros(input.size(0), 51), requires_grad=False).cuda()
            c_t = Variable(torch.zeros(input.size(0), 51), requires_grad=False).cuda()
            h_t2 = Variable(torch.zeros(input.size(0), 51), requires_grad=False).cuda()
            c_t2 = Variable(torch.zeros(input.size(0), 51), requires_grad=False).cuda()
        else:
            h_t = Variable(torch.zeros(input.size(0), 51), requires_grad=False)
            c_t = Variable(torch.zeros(input.size(0), 51), requires_grad=False)
            h_t2 = Variable(torch.zeros(input.size(0), 51), requires_grad=False)
            c_t2 = Variable(torch.zeros(input.size(0), 51), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

def train(x, y, ret_all_losses=False):
    model.train()
    x, y = torch_data(x, y)
    data, target = to_var(x), to_var(y)
    optimizer.zero_grad()
    output = model(data)

    all_losses = None
    if ret_all_losses:
        all_losses = F.mse_loss(output, target, reduce=False)
        if args.cuda:
            all_losses = all_losses.cpu().data.numpy()
        else:
            all_losses = all_losses.data.numpy()

    loss = F.mse_loss(output, target)
    val_loss = loss.data[0]/x.shape[0]
    loss.backward()
    optimizer.step()
    return val_loss, all_losses

def test(x, y):
    model.eval()
    test_loss = 0

    x, y = torch_data(x, y)
    data, target = to_var(x, volatile=True), to_var(y, volatile=True)

    future = 1000
    output = model(data, future=future)
    test_loss = F.mse_loss(output[:, :-future], target, size_average=False)
    if args.cuda:
        return test_loss.cpu().data.numpy()[0]/float(x.shape[0])
    else:
        return test_loss.data.numpy()[0]/float(x.shape[0])
    # test_loss /= x_test.shape[0]
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        # test_loss, correct, x_test.shape[0],
        # 100. * correct / x_test.shape[0]))

def bernoulli_sample(score, sample_size):
    sel = np.random.binomial(1, prob)
    return sel


def gradients(x, y):
    n, f = x.shape
    x, y = torch_data(x, y)
    data, target = to_var(x), to_var(y)

    weight_grads = np.zeros((n,))
    for i in np.arange(n):
        output = model(data[i].view(-1, f))
        loss = F.mse_loss(output, target[i])
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
        loss_batch = np.mean(loss_batch, axis=1)
        losses[:, 0][:size] = loss_batch
        objective_epoch.append(loss)
    return objective_epoch, losses


if args.dataset == 'mnist':
    model = MNIST_Net()
elif args.dataset == 'cifar':
    model = CIFAR_Net()
elif args.dataset == 'sine':
    model = Sequence()

if args.cuda:
    model.cuda()

x_train, y_train, x_test, y_test, x_val, y_val = load_dataset(args.dataset)
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
    losses = np.mean(losses, axis=1)
    # losses = losses.reshape(losses.shape[0], 1)
    losses = np.column_stack([losses, range(losses.shape[0])])
    for epoch in np.arange(args.epochs-1):
        objective_epoch, losses = objective_sampling(x_train, y_train, losses)
        test(x_test, y_test)
elif args.sample_type == 'var':
    # x_train_var = np.var(x_train.reshape(x_train.shape[0], reshape_size), axis=1)
    x_train_var = np.var(x_train, axis=1)
    prob = args.batch_size * (x_train_var / np.sum(x_train_var))
    for epoch in np.arange(args.epochs):
        variance_sampling(x_train, y_train, prob)
        test(x_test, y_test)
elif args.sample_type == 'uni':
    for epoch in np.arange(args.epochs):
        uniform_sampling(x_train, y_train)
        test(x_test, y_test)
