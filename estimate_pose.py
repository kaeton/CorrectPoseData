import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

class EstimatePoseMovie:
    def __init__(self):
        self.feature = {}

    # def mk_feature(self, label: str, input_movie: str) -> None:
    def mk_feature(self, label, input_movie):
        # self.src_movie_title = input_movie
        src_video = cv2.VideoCapture(input_movie)
        feature = []

        ret, frame = src_video.read()
        while ret is True:
            feature.append(frame)
            ret, frame = src_video.read()

        self.feature[label] = feature

    def training(self, use_feature, cross_validation):
        self.train_data = []
        self.label_data = []


        for label in use_feature:
            self.label_data.extend([label for x in self.feature[label]])
            self.train_data.extend(self.feature[label])

        print("shape", self.train_data)
        print("shape", self.label_data)

        clf = svm.SVC(kernel='linear', C=1)
        scores = cross_val_score(clf, self.train_data, self.label_data, cv=cross_validation)
        print("scores", scores)


if __name__ == "__main__":
    estimator = EstimatePoseMovie()
    estimator.mk_feature("sit", "mp4")

    estimator.training(use_feature=["sit", "raise_hands"], cross_validation=3)
