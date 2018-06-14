import cv2
import numpy as np
from sklearn import svm
import yaml
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import cohen_kappa_score
from sklearn import neural_network
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import itertools


class EstimatePoseMovie:
    def __init__(self):
        self.feature = {}
        self.uselabel = []

    # def mk_feature(self, label: str, input_movie: str) -> None:
    def mk_feature(self, label, input_movie):
        # self.src_movie_title = input_movie
        self.uselabel.append(label)
        src_video = cv2.VideoCapture(input_movie)
        feature = []

        # フレーム画像サイズをリサイズしてもいいかも
        ret, frame = src_video.read()
        while ret is True:
            frame = cv2.resize(frame,(128, 96))
            print(np.shape(frame))
            # frame = np.reshape(frame, (921600,))
            frame = np.reshape(frame, (36864,))
            feature.append(frame)

            # for i in range(10):
            ret, frame = src_video.read()

        self.feature[label] = feature

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def cross_validation(self, use_feature, cross_validation, hidden_neuron):
        self.train_data = []
        self.label_data = []
        for x, label in enumerate(use_feature):
            self.label_data.extend([x for i in self.feature[label]])
            self.train_data.extend(self.feature[label])
            print([x for i in self.feature[label]])
            print(np.shape(self.feature[label]))

        print("shape", np.shape(self.train_data))
        print("shape", np.shape(self.label_data))
        print("shape", self.label_data)
        # kernel='rbf', C=1
        # clf = svm.SVC(kernel='rbf', C=200)
        clf = neural_network.MLPClassifier(activation="relu", hidden_layer_sizes=hidden_neuron)
        # print(cv)

        y_pred = cross_val_predict(clf, self.train_data, self.label_data, cv= KFold(n_splits=10, shuffle=True))
        conf_mat = confusion_matrix(self.label_data, y_pred)
        self.plot_confusion_matrix(conf_mat, self.label_data)
        # accuracy = cohen_kappa_score(self.label_data, y_pred)
        # conf_mat = confusion_matrix(self.label_data, y_pred)
        # print("the result of neural network")
        # print("accuracy", accuracy)
        # print("conf_mat", conf_mat)


if __name__ == "__main__":
    f = open("setting.yaml", "r+")
    setting = yaml.load(f)
    print(setting)

    label_list = []

    estimator = EstimatePoseMovie()
    for i in setting["filename"]:
        # print(i, setting["filename"][i])
        label_list.append(i)
        estimator.mk_feature(i, setting["filename"][i])

    # estimator.mk_feature("30bpm", "../bone_clapping_motion/bpm_30_0.mp4")
    # estimator.mk_feature("60bpm", "../bone_clapping_motion/bpm_60_0.mp4")
    # estimator.mk_feature("90bpm", "../bone_clapping_motion/bpm_90_0.mp4")
    # estimator.mk_feature("120bpm","../bone_clapping_motion/bpm_120_0.mp4")
    # estimator = EstimatePoseMovie()
    # estimator.mk_feature("sit", "../experiments_data/bone_picture/sit_0.mp4")
    # estimator.mk_feature("t_pose", "../experiments_data/bone_picture/t_pose_0.mp4")
    # # estimator.mk_feature("raise_hands", "../experiments_data/bone_picture/one_position/raise_hands_re_0.mp4")
    # estimator.mk_feature("walking", "../experiments_data/bone_picture/background_walking_bone_0.mp4")

    #estimator.cross_validation(use_feature=["sit", "t_pose", "raise_hands", "walking"], cross_validation=3)
    for neuron in setting["hidden_layer"]:
        estimator.cross_validation(use_feature=label_list, cross_validation=3, hidden_neuron=neuron)

