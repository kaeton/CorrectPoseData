from estimate_pose import EstimatePoseMovie
from posedata_loader import PosedataLoader
import yaml
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2


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
# clf = DBSCAN(eps=1.2, min_samples=2)
clf = KMeans(n_clusters=4, random_state=10)
result = clf.fit(list(feature_df["feature_1d"]))
result.get_params()
y_dbscan = result.labels_
y_dbscan

x_dbscan = result.cluster_centers_
# x_dbscan = result.components_
x_dbscan
matrix_result = confusion_matrix(feature_df["label"], y_dbscan)
matrix_result
matrix_result = np.array(matrix_result, dtype='int64')
np.shape(matrix_result)
np.savetxt("out_matrix.csv", matrix_result, delimiter=",")

for result_label, image in zip(y_dbscan, feature_df["feature_2d"]):
    print(image)
    print(result_label)
    cv2.imshow(winname=str(result_label), mat=np.array(image))
    key = cv2.waitKey()
    if key is "q":
        break
    cv2.destroyAllWindows()

image = feature_df["feature_2d"][0]
cv2.imshow("test", image)
cv2.waitKey()
cv2.destroyAllWindows()
