from __future__ import print_function
import argparse
import torch
import sys
import os
import struct
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from multiprocessing.dummy import  Pool as ThreadPool
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def load_dataset(path):

    # training data
    data = [np.load(os.path.join(path, 'cifar-10-batches-py',
                                 'data_batch_%d' % (i + 1))) for i in range(5)]
    X_train = np.vstack([d['data'] for d in data])
    y_train = np.hstack([np.asarray(d['labels'], np.int8) for d in data])

    # test data
    data = np.load(os.path.join(path, 'cifar-10-batches-py', 'test_batch'))
    X_test = data['data']
    y_test = np.asarray(data['labels'], np.int8)

    # reshape
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    # normalize
    try:
        mean_std = np.load(os.path.join(path, 'cifar-10-mean_std.npz'))
        mean = mean_std['mean']
        std = mean_std['std']
    except IOError:
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = X_train.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        np.savez(os.path.join(path, 'cifar-10-mean_std.npz'),
                 mean=mean, std=std)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


# def model_loss(x_train, y_train):
#     x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
#     y_train = torch.from_numpy(y_train).type(torch.LongTensor)
#     if args.cuda:
#         data, target = x_train.cuda(), y_train.cuda()
#     else:
#         data, target = x_train, y_train
#     data, target = Variable(data), Variable(target)
#     optimizer.zero_grad()
#     output = model(data)
#     loss = F.cross_entropy(output, target, reduce=False)
#     losses = loss.cpu().data.numpy()
#     # losses = np.zeros(x_train.shape[0])
#     loss = F.nll_loss(output, target)
#
#     return loss


def train2(x_train, y_train):
    model.train()
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    if args.cuda:
        data, target = x_train.cuda(), y_train.cuda()
    else:
        data, target = x_train, y_train
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target, reduce=False)
    losses = loss.cpu().data.numpy()
    # losses = np.zeros(x_train.shape[0])
    loss = F.nll_loss(output, target)
    # loss = loss.sum()
    val_loss = loss.data[0]
    loss.backward()
    optimizer.step()
    # print(loss.data)
    return val_loss

# def train(x_train, y_train):
    # model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
        # if args.cuda:
            # data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        # optimizer.zero_grad()
        # output = model(data)
        # # criterion = nn.CrossEntropyLoss()
        # loss = F.cross_entropy(output, target, reduce=False)
        # # loss = criterion(output, target, reduce=False)
        # losses[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = loss.data.numpy()
        # loss = loss.sum()
        # loss.backward()
        # optimizer.step()
        # if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                # epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.data[0]))


def test2(x_test, y_test):
    model.eval()
    test_loss = 0
    correct = 0
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    if args.cuda:
        data, target = x_test.cuda(), y_test.cuda()
    else:
        data, target = x_test, y_test
    data, target = Variable(data, volatile=True), Variable(target, volatile=True)
    output = model(data)
    test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= 10000
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, 10000,
        100. * correct / 10000))

# def test():
    # model.eval()
    # test_loss = 0
    # correct = 0
    # for data, target in test_loader:
        # if args.cuda:
            # data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)
        # test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # test_loss, correct, len(test_loader.dataset),
        # 100. * correct / len(test_loader.dataset)))


def parallel_fun(data, target, i):
    print(i)
    output = model(data[i].view(1, 1, 28, 28))
    loss = F.nll_loss(output, target[i])
    loss.backward()

    model_params = list(model.parameters())
    norm = 0
    for param in model_params:
        grad = param.grad.data.numpy()
        norm = np.sqrt((grad * grad).sum() + (norm * norm))

    return norm


def weight_grads(x_train, y_train):
    N = x_train.shape[0]

    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    if args.cuda:
        data, target = x_train.cuda(), y_train.cuda()
    else:
        data, target = x_train, y_train
    data, target = Variable(data), Variable(target)

    grads = np.zeros((N,))

    # loss = F.cross_entropy(output, target, reduce=False)
    # losses = loss.cpu().data.numpy()

    # loss = F.nll_loss(output, target)
    # loss.backward()
    # optimizer.step()

    num_cores = multiprocessing.cpu_count()
    print(num_cores)

    # results = Parallel(n_jobs=num_cores)(delayed(parallel_fun)(data=data, target=target, i=i) for i in np.arange(N))

    # pool = ThreadPool(num_cores)
    # results = pool.map(parallel_fun, )

    for i in np.arange(N):
        output = model(data[i].view(1, 3, 32, 32))
        loss = F.nll_loss(output, target[i])
        loss.backward()

        model_params = list(model.parameters())
        norm = 0
        for param in model_params:
            if args.cuda:
                grad = param.grad.cpu().data.numpy()
            else:
                grad = param.grad.data.numpy()
            norm = np.sqrt((grad * grad).sum() + (norm * norm))

        grads[i] = norm
        optimizer.zero_grad()

    print(grads)
    return grads

n = 50000
# losses = torch.Tensor(50000).zero_()

# x_train = read_idx('../data/raw/train-images-idx3-ubyte')
# x_test = read_idx('../data/raw/t10k-images-idx3-ubyte')
# y_train = read_idx('../data/raw/train-labels-idx1-ubyte')
# y_test = read_idx('../data/raw/t10k-labels-idx1-ubyte')

# x_train = x_train.reshape(60000, 1, 28, 28)
# x_test = x_test.reshape(10000, 1, 28, 28)

x_train, y_train, x_test, y_test = load_dataset('../data')
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# print(torch.Tensor(x_train).size(0))
# print(torch.Tensor(y_train).size(0))

# train  = data_utils.TensorDataset(x_train, y_train)
# train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

# for val in train_loader:
    # print(val)


x_train = x_train[:n]
y_train = y_train[:n]

#losses = train2(x_train, y_train)
#losses = losses.reshape(losses.shape[0], 1)
#losses = np.column_stack([losses, range(losses.shape[0])])

# losses = weight_grads(x_train, y_train)
# losses = losses.reshape(losses.shape[0], 1)
# losses = np.column_stack([losses, range(losses.shape[0])])

losses = np.zeros((n, 2))
uniform = []
for epoch in range(50):
    uniform_epoch = []
    for i in range(n/args.batch_size):

        # ind = np.argsort(losses[:, 0])
        # losses = losses[ind][::-1]
        # idx = losses[:, 1][:64]
        # losses[:, 0][:64] = losses[:, 0][:64]/50;
        # idx = idx.astype(int)
        # loss_batch = train2(x_train[idx], y_train[idx])
        uniform_epoch.append(train2(x_train[i*64:(i+1)*64], y_train[i*64:(i+1)*64]))
        # loss_batch = train2(x_train, y_train)
        # losses[:, 0][:64] = loss_batch
    uniform.append(uniform_epoch)
    print('Uniform Train Loss Mean', np.mean(uniform_epoch))
    test2(x_test, y_test)

uniform = np.array(uniform)
uniform = np.mean(uniform, axis=0)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
non_uniform = []
for epoch in range(3):
    non_uniform_epoch = []
    for i in range(n/args.batch_size):

        # ind = np.argsort(losses[:, 0])
        # losses = losses[ind][::-1]
        # idx = losses[:, 1][:64]
        # losses[:, 0][:64] = losses[:, 0][:64]/50;
        # idx = idx.astype(int)
        # loss_batch = train2(x_train[idx], y_train[idx])
        non_uniform_epoch.append(train2(x_train[i*64:(i+1)*64], y_train[i*64:(i+1)*64]))
        # loss_batch = train2(x_train, y_train)
        # losses[:, 0][:64] = loss_batch
    non_uniform.append(non_uniform_epoch)
    print('Non Uniform Train Loss Mean', np.mean(non_uniform_epoch))
    test2(x_test, y_test)

losses = weight_grads(x_train, y_train)
losses = losses.reshape(losses.shape[0], 1)
losses = np.column_stack([losses, range(losses.shape[0])])
#print (np.var(losses[:, 0]), np.mean(losses[:, 0]), np.max(losses[:, 0]), np.min(losses[:, 0]))

for epoch in range(47):
    non_uniform_epoch = []
    count = np.zeros(n)
    for i in range(n/args.batch_size):
        ind = np.argsort(losses[:, 0])
        losses = losses[ind][::-1]
        idx = losses[:, 1][:64]
        losses[:, 0][:64] = losses[:, 0][:64]/1.5

        idx = idx.astype(int)
        count[idx] +=1
        loss_batch = train2(x_train[idx], y_train[idx])
        non_uniform_epoch.append(loss_batch)
        # loss_batch = train2(x_train[i*64:(i+1)*64], y_train[i*64:(i+1)*64])
        # loss_batch = train2(x_train, y_train)
        # losses[:, 0][:64] = loss_batch
    plt.clf()
    print(np.var(count))
    plt.scatter(range(n),count)
    plt.savefig('count'+str(epoch)+'.png')
    non_uniform.append(non_uniform_epoch)
    print('Non Uniform Train Loss Mean', np.mean(non_uniform_epoch))
    test2(x_test, y_test)

non_uniform = np.array(non_uniform)
non_uniform = np.mean(non_uniform, axis=0)
plt.clf()
plt.plot(uniform[1:], label='uni')
plt.plot(non_uniform[1:], label='nuni')
plt.legend()
plt.savefig('compare.png')

# norm_x = ((x_train.transpose(3,2,1,0) - np.mean(x_train.reshape(-1, 784), axis=1)) / np.var(x_train.reshape(-1, 784), axis=1)).transpose(3,2,1,0)
