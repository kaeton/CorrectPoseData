#!usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.transforms as transforms
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# データセットをダウンロード
mnist_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
data_loader = DataLoader(mnist_data,
                        batch_size=4,
                        shuffle=False)

# 中身を見てみる
data_iter = iter(data_loader)
images, labels = data_iter.next()

train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_data,
                        batch_size=4,
                        shuffle=True)
test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data,
                         batch_size=4,
                         shuffle=False)


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 42)
        self.fc3 = nn.Linear(42, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.log_softmax(self.fc3(x))
        x = F.softmax(self.fc3(x))
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Linear(28 * 28, 50)
#         self.l2 = nn.Linear(50, 10)
#                        
#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = self.l1(x)
#         x = self.l2(x)
#         y = nn.functional.log_softmax(x)
#         return y

net = Net()

# コスト関数と最適化手法を定義
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        # Variableに変換
        inputs, labels = Variable(inputs), Variable(labels)
        
        # 勾配情報をリセット
        optimizer.zero_grad()
        
        # 順伝播
        outputs = net(inputs)
        if i % 5000 == 4999:
            print(labels)
            print(outputs)
        
        # ロスの計算
        loss = criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        
        # パラメータの更新
        optimizer.step()
        
        running_loss += loss.data[0]
        
        if i % 5000 == 4999:
            print('%d %d loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0



print('Finished Training')
torch.save(net.state_dict(), 'weight.pth')
