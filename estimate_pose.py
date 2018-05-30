import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import cohen_kappa_score
from sklearn import neural_network

class EstimatePoseMovie:
    def __init__(self):
        self.feature = {}

    # def mk_feature(self, label: str, input_movie: str) -> None:
    def mk_feature(self, label, input_movie):
        # self.src_movie_title = input_movie
        src_video = cv2.VideoCapture(input_movie)
        feature = []

        # フレーム画像サイズをリサイズしてもいいかも
        ret, frame = src_video.read()
        while ret is True:
            # frame_resized = cv2.resize(frame,(128, 96))
            frame = np.reshape(frame, (921600,))
            feature.append(frame)
            for i in range(10):
                ret, frame = src_video.read()

        self.feature[label] = feature

    def cross_validation(self, use_feature, cross_validation):
        self.train_data = []
        self.label_data = []

        for x, label in enumerate(use_feature):
            self.label_data.extend([x for i in self.feature[label]])
            self.train_data.extend(self.feature[label])

        print("shape", np.shape(self.train_data))
        print("shape", np.shape(self.label_data))
        print("shape", self.label_data)
        # kernel='rbf', C=1
        clf = svm.SVC(kernel='rbf', C=1)
        # clf = neural_network.MLPClassifier(activation="relu")
        cv = ShuffleSplit(n_splits=cross_validation, test_size=0.3, random_state=0)
        # print(cv)

        # ここでのシャッフルの命令がcross_val_predictではできていなかった可能性が高い。

        score = cross_val_score(clf, self.train_data, self.label_data, cv=cv)
        print(score)
        # y_pred = cross_val_predict(clf, self.train_data, self.label_data, cv=cross_validation)
        # accuracy = cohen_kappa_score(self.label_data, y_pred)
        # conf_mat = confusion_matrix(self.label_data, y_pred)
        # print("the result of neural network")
        # print("accuracy", accuracy)
        # print("conf_mat", conf_mat)


if __name__ == "__main__":
    estimator = EstimatePoseMovie()
    estimator.mk_feature("sit", "../experiments_data/bone_picture/sit_0.mp4")
    estimator.mk_feature("t_pose", "../experiments_data/bone_picture/t_pose_0.mp4")
    estimator.mk_feature("raise_hands", "../experiments_data/bone_picture/raise_hands_re_0.mp4")
    #estimator.cross_validation(use_feature=["sit", "t_pose", "raise_hands", "walking"], cross_validation=3)
    estimator.cross_validation(use_feature=["sit", "t_pose", "raise_hands"], cross_validation=3)

    second_person = EstimatePoseMovie()

