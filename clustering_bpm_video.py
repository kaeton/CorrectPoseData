from estimate_pose import EstimatePoseMovie
from posedata_loader import PosedataLoader
import yaml
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from gmm import Model
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os


label_list = []

f = open("setting_train_test.yaml", "r+")
setting = yaml.load(f)
print(setting)
# extension_instance = RectangularExtraction(resizesize=(30, 60), offset=15)
extension_instance = EstimatePoseMovie()

dataloader = PosedataLoader()

label = "120bpm"
for filepath in setting["test"][label]:
    # estimator.mk_feature_from_moviefile(label, filepath)
    print("label filepath", label, filepath)
    dataloader.extend_frame_by_label(
        label=label,
        movie_src=filepath,
        table_src=filepath + ".csv"
    )

# label, data = dataloader.translate(
feature_df = dataloader.translate(
    batchsize=None,
    random_state=1,
    do_shuffle=False,
    use_label=[label]
)

label_np = np.array(label)
examdirname = "result_classification/exam1102_dbscan_1.0/"
clf = DBSCAN(eps=0.9, min_samples=2)
# clf = KMeans(n_clusters=2, random_state=10)
# clf = GaussianMixture(
#     n_components=2,
#     covariance_type='full',
#     random_state=5
# )
# gaussian mixture model test
##############################################
# clf = Model()
# result = clf.fit(
#     X=list(feature_df["feature_1d"]),
#     do_show=False
# )
# y_pred = result
##############################################

result = clf.fit(list(feature_df["feature_1d"]))
# y_pred = result.predict(list(feature_df["feature_1d"]))
y_pred = result.labels_

matrix_result = confusion_matrix(feature_df["label"], y_pred)
matrix_result
# matrix_result = np.array(matrix_result, dtype='int64')
# np.shape(matrix_result)
# np.savetxt("out_matrix.csv", matrix_result, delimiter=",")

label_list = list(set(y_pred))
label_list
for label in label_list:
    os.makedirs(examdirname + str(label))

i = 0
for result_label, image in zip(y_pred, feature_df["feature_2d"]):
    print(image)
    print(result_label)
    # plt.title(str(result_label))
    # plt.imshow(np.array(image))
    # plt.show()

    print(examdirname + str(label) + "/" + str(i) + ".png")
    cv2.imwrite(examdirname + str(result_label) + "/" + str(i) + ".png", 255 * np.array(image))
    i += 1
    # cv2.imshow(winname=str(result_label), mat=np.array(image))
    # key = cv2.waitKey()
    # print("key", key)
    # if key is "q":
    #     break
    # cv2.destroyAllWindows()
