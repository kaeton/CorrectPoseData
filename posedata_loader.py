import torch
import numpy as np
import pandas as pd
from estimate_pose import EstimatePoseMovie

# this class prepare tensor data for pytorch

class PosedataLoader:
    def __init__(self):
        self.movie_data = []
        self.label_data = []

    # numpy配列からtensor dataへ変換する
    def translate(self, batchsize):

    # 動画ファイルを読み込んで、行列化する
    def load_movie(self, label, src):
        mk_movie_data = EstimatePoseMovie()
        frame_data = mk_movie_data.\
            mk_feature_from_moviefile(
            label=label,
            input_movie=src,
            reshape=False
        )
        self.movie_data.append(frame_data)

    # csvを読み込んで行列化する
    def load_csv(self, src):
        label = pd.read_csv(src + ".csv")
        self.label_data.append(label)

    # 動画行列をcsvのラベルごとに振り分ける
    def distribute_frame_by_label(self):
