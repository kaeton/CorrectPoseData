import torchvision.transforms as transforms
import torch
from torch.autograd import Variable

import torch.optim as optim
from posedata_loader import PosedataLoader
from estimate_pose import EstimatePoseMovie
from rectangular_extraction import RectangularExtraction
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import yaml

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30 * 60, 50)
        self.fc2 = nn.Linear(50, 42)
        self.fc3 = nn.Linear(42, 10)

    def forward(self, x):
        x = x.view(-1, 30 * 60)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.log_softmax(self.fc3(x))
        x = F.softmax(self.fc3(x))
        return x

if __name__ == "__main__":
    f = open("setting.yaml", "r+")
    setting = yaml.load(f)
    print(setting)
    # extension_instance = RectangularExtraction(resizesize=(30, 60), offset=15)
    extension_instance = EstimatePoseMovie()

    label_list = []
    dataloader = PosedataLoader()

    for label in setting["filename"]:
        # print(i, setting["filename"][i])
        label_list.append(label)
        for filepath in setting["filename"][label]:
            # estimator.mk_feature_from_moviefile(label, filepath)
            print("label filepath", label, filepath)
            dataloader.extend_frame_by_label(
                label=label,
                movie_src=filepath,
                table_src=filepath + ".csv"
            )

    # dataloader.get()
    train_loader = dataloader.translate(batchsize=4, use_label=["30bpm", "60bpm", "90bpm"])
    test_loader = np.array(dataloader.mk_movie_data.feature["120bpm"])
    test_loader = extension_instance.mk_feature_humanextraction_array(framearray=test_loader)
    print(test_loader[0])

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # Variableに変換
            # inputs = torch.from_numpy(inputs)
            # labels = torch.from_numpy(labels)
            inputs = torch.FloatTensor(inputs)
            labels = torch.LongTensor(labels)

            # 勾配情報をリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = net(inputs)
            if i % 100 == 99:
                print(labels)
                print(outputs)

            # ロスの計算
            loss = criterion(outputs, labels)

            # 逆伝播
            loss.backward()

            # パラメータの更新
            optimizer.step()

            running_loss += loss.data[0]

            if i % 100 == 99:
                print('%d %d loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    # torch.save(net.state_dict(), "weight.pth")
    outputs = net(test_loader)
    print(outputs)
