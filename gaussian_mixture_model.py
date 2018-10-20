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


