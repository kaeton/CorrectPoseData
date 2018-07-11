import torchvision.transforms as transforms
import torch
from torch.autograd import Variable

import torch.optim as optim
from posedata_loader import PosedataLoader
from estimate_pose import EstimatePoseMovie
from rectangular_extraction import RectangularExtraction
import numpy as np
import cv2

import torch.nn as nn
import torch.nn.functional as F
import yaml

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30 * 60, 42)
        # self.fc2 = nn.Linear(50, 42)
        self.fc3 = nn.Linear(42, 2)

    def forward(self, x):
        x = x.view(-1, 30 * 60)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.log_softmax(self.fc3(x))
        x = F.softmax(self.fc3(x))
        return x

if __name__ == "__main__":
    f = open("setting_eastside.yaml", "r+")
    setting = yaml.load(f)
    print(setting)
    # extension_instance = RectangularExtraction(resizesize=(30, 60), offset=15)
    extension_instance = EstimatePoseMovie()

    dataloader = PosedataLoader()

    for label in setting["train"]:
        # print(i, setting["filename"][i])
        for filepath in setting["train"][label]:
            # estimator.mk_feature_from_moviefile(label, filepath)
            print("label filepath", label, filepath)
            dataloader.extend_frame_by_label(
                label=label,
                movie_src=filepath,
                table_src=filepath + ".csv"
            )

    for label in setting["test"]:
        # print(i, setting["filename"][i])
        for filepath in setting["test"][label]:
            # estimator.mk_feature_from_moviefile(label, filepath)
            print("label filepath", label, filepath)
            dataloader.extend_frame_by_label(
                label=label,
                movie_src=filepath,
                table_src=filepath + ".csv"
            )

    # dataloader.get()
    # test_loader = np.array(dataloader.mk_movie_data.feature["120bpm"])
    # test_loader = extension_instance.mk_feature_humanextraction_array(framearray=test_loader)
    # print(test_loader[0])

    # test_loader = dataloader.translate(batchsize=None, use_label=["120bpm"])
    # test_loader = dataloader.mk_movie_data.mk_feature_humanextraction_array(
    #     use_gray_image=True,
    #     framearray=
    # )

    _, test_loader = dataloader.translate(
        batchsize=None,
        random_state=None,
        use_label=["120bpm"]
    )
    print(np.shape(test_loader))

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(10):
        print(epoch, "epoch")
        running_loss = 0.0
        train_loader = dataloader.translate(batchsize=4, random_state=epoch, use_label=["open", "close"])
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # Variableに変換
            # inputs = torch.from_numpy(inputs)
            # labels = torch.from_numpy(labels)

            # for label, frame in zip(labels, inputs):
            #     print(type(frame))
            #     cv2.imshow("data", frame)
            #     cv2.waitKey()

            inputs = torch.FloatTensor(inputs)
            labels = torch.LongTensor(labels)

            # 勾配情報をリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = net(inputs)
            if i % 100 == 99:
                print("inputs", inputs)
                print("outputs", outputs)
                print("labels", labels)

                # for label, frame in zip(labels, inputs):
                #     cv2.imshow("data" + label, frame)
                #     cv2.waitKey()

                # cv2.destroyAllWindows()

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

    for label in setting["test"]:
        # print(i, setting["filename"][i])
        for filepath in setting["test"][label]:
            print("label filepath", label, filepath)

            _, test_loader = dataloader.translate(
                batchsize=None,
                random_state=None,
                use_label=[label]
            )
            # torch.save(net.state_dict(), "weight.pth")
            print("test loader shape", np.shape(test_loader))
            test_loader = torch.FloatTensor(test_loader)
            outputs = net(test_loader)
            np.savetxt(label + "result.csv", outputs.detach().numpy(), delimiter=',')
