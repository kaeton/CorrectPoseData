import numpy as np
import yaml
import cv2
from estimate_pose import EstimatePoseMovie
from sklearn.utils import shuffle

# this class prepare tensor data for pytorch

class PosedataLoader:
    def __init__(self):
        self.movie_data = []
        self.label_data = {}
        self.mk_movie_data = EstimatePoseMovie()

    # numpy配列からtensor dataへ変換する
    # TODO ; もっと細かく区切る
    # this batch size for training data by pytorch
    def translate(self, batchsize=4, random_state=0, use_label=[]):
        print("use_labael", use_label)
        feature_arr = []
        label_arr = []

        for i in use_label:
            print("label data", i)
            print("label array shape", np.shape(self.label_data[i]))
            print("data array shape", np.shape(self.mk_movie_data.feature[i]))

            length = np.shape(self.label_data[i])[0]
            # feature_use = self.mk_movie_data.feature[i][:length]
            # print("feature use", np.shape(feature_use))
            # for frame, label in zip(
            #         self.mk_movie_data.feature[i][:length],
            #         self.label_data[i]
            # ):

            feature_arr.extend(self.mk_movie_data.feature[i][:length])
            print("label array shape", np.shape(self.label_data[i]))
            label_arr.extend(self.label_data[i])

        use_feature = [feature_arr[i] for i, label in enumerate(label_arr) if label >= 0]
        use_label = [i for i in label_arr if i >= 0]
        use_label, use_feature =\
            self.mk_movie_data.mk_feature_humanextraction_array(
                labelarray=use_label,
                framearray=use_feature,
                use_gray_image=True
            )

        use_feature, use_label = shuffle(use_feature, use_label, random_state=random_state)
        if batchsize is None:
            return [use_label, use_feature]
        else:
            return self.mk_batchdata(
                feature=use_feature,
                label=use_label,
                batchsize=batchsize
            )

    # image data arrayを1d変換しておく
    def reshape(self, data, size=(1800,)):
        reshape_data = []
        for image in data:
            reshaped = np.reshape(image, size)
            reshape_data.append(reshaped)

        return reshape_data

    def mk_batchdata(self, feature, label, batchsize):
        start_position_each_data = \
            [i for i in range(0, np.shape(feature)[0], batchsize)]
        batch_train_data = \
            [np.array(feature[p:][:batchsize]) for p in start_position_each_data]
        batch_label_data = \
            [np.array(label[p:][:batchsize]) for p in start_position_each_data]
        # tensor_batch_train_data = \
        #     [data for data in batch_train_data if len(data) == batchsize]
        # tensor_batch_label_data = \
        #     [data for data in batch_label_data if len(data) == batchsize]

        tensor_data = \
            [[batchdata, batchlabel] \
             for batchdata, batchlabel in zip(batch_train_data, batch_label_data)
             if len(batchdata) == batchsize and len(batchlabel) == batchsize]

        # tensor_data = \
        #     [[torch.from_numpy(batchdata), torch.from_numpy(batchlabel)] \
        #      for batchdata, batchlabel in zip(batch_train_data, batch_label_data)
        #      if np.shape(batchdata)[0] == batchsize and np.shape(batchlabel)[0] == batchsize]
        # print(np.shape(tensor_data))
        # print(tensor_data[0])

        return tensor_data

    # 画像データとラベルの確認用
    def check_feature(self, feature, label):
        for frame, l in zip(feature, label):
            cv2.imshow("label : " + str(l), frame)
            cv2.waitKey()

        cv2.destroyAllWindows()


    def get(self):
        for i in self.mk_movie_data.feature:
            print(i, np.shape(self.mk_movie_data.feature[i]))
        print(len(self.label_data))

    # 動画ファイルを読み込んで、行列化する
    # label データの定義用、　30bpmとか
    # src 動画データのパス
    # TODO : reshape = Falseとしているが、変数名が短すぎて流石に複雑すぎる。また引数も必要とするべき
    # reshape optionはそのまま学習する際に画像データを一次元に変換しておくもの。
    # 今回はまだ処理が残っているので、データを抽出用に使用するため、reshape はFalseとなる
    def load_movie(self, label, src):
        self.mk_movie_data.mk_feature_from_moviefile(
            label=label,
            input_movie=src,
            reshape=False
        )
        # self.movie_data.append(frame_data)


    # csvを読み込んで行列化する
    def load_csv(self, src, label):
        label_table = np.loadtxt(src)
        self.label_data[label] = label_table

    # frameを読み込み、１ファイルに格納する
    def extend_frame_by_label(self, label, movie_src, table_src):
        self.load_movie(label=label, src=movie_src)
        self.load_csv(label=label, src=table_src)

if __name__ == "__main__":
    f = open("setting_train_test.yaml", "r+")
    setting = yaml.load(f)
    print(setting)

    label_list = []
    dataloader = PosedataLoader()
    print(dataloader.__dir__())

    for label in setting["train"]:
        # print(i, setting["filename"][i])
        label_list.append(label)
        for filepath in setting["train"][label]:
            # estimator.mk_feature_from_moviefile(label, filepath)
            print("label filepath", label, filepath)
            dataloader.extend_frame_by_label(
                label=label,
                movie_src=filepath,
                table_src=filepath + ".csv"
            )

    # dataloader.get()
    data = dataloader.translate(batchsize=None, use_label=["open", "close"])
    for image in data[:2]:
        print("123")
        # print(image)
        # print(type(image))


        # cv2.imshow("image %d".format(str(x)), image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    # dataloader.extend_frame_by_label(src=)
