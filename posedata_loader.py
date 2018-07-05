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
    # TODO: actionのラベルをリストに入れておくことで、どのaction labelを使うのかを定義する
    def translate(self, batchsize=4, use_label=[]):
        feature_arr = []
        label_arr = []

        for i in use_label:
            print("label data", i)
            print("label array shape", np.shape(self.label_data[i]))
            print(self.label_data[i])
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

        # np.savetxt("flag1.csv", label_arr)
        use_feature = [feature_arr[i] for i, label in enumerate(label_arr) if label > 0 and label != 1]
        use_label = [i for i in label_arr if i > 0 and i != 2]
        # np.savetxt("flag2.csv", use_label_)
        # use_label = [i-1 for i in use_label_]
        # np.savetxt("flag3.csv", use_label)
        use_label, use_feature =\
            self.mk_movie_data.mk_feature_humanextraction_array(
                labelarray=use_label,
                framearray=use_feature,
                use_gray_image=True
            )
        print("use feature length", np.shape(use_feature))
        print("use feature length", np.shape(use_label))
        use_feature, use_label = shuffle(use_feature, use_label, random_state=0)

        print("use feature length", np.shape(use_feature))
        print("use feature length", np.shape(use_label))

        return self.mk_batchdata(
            feature=use_feature,
            label=use_label,
            batchsize=batchsize
        )

        # self.check_feature(feature=use_feature, label=use_label)
        # for i, label in label_arr:
        #     if i <= 0:

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
    f = open("setting.yaml", "r+")
    setting = yaml.load(f)
    print(setting)

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
    dataloader.translate(batchsize=4, use_label=["30bpm", "60bpm"])

    # dataloader.extend_frame_by_label(src=)
