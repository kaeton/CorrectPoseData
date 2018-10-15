from estimate_pose import EstimatePoseMovie
from posedata_loader import PosedataLoader
import yaml
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np


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

label, data = dataloader.translate(
    batchsize=None,
    random_state=1,
    use_label=[label]
)
label_np = np.array(label)
# print(np.shape(np.array(train_loader)))
# data_2d = dataloader.reshape_1D(data)
data_2d = dataloader.reshape(data)
# clf = DBSCAN(eps=1.5, min_samples=2)
clf = KMeans(n_clusters=2, random_state=10)
result = clf.fit(data_2d)
result.get_params()
y_dbscan = result.labels_
matrix_result = confusion_matrix(label, y_dbscan)
matrix_result
