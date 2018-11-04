import os
import numpy as np
import cv2

# examdirname : 実験の結果を記録するディレクトリ
# （result_classificationの中にディレクトリが作成されるようになる
# y_pred : classificationの結果のラベルが入ったリスト
#
def save_clustering_image(examdirname, cluster_label, cluster_image):
    label_list = list(set(cluster_label))
    for label in label_list:
        os.makedirs(examdirname + str(label))

    i = 0
    for result_label, image in zip(cluster_label, cluster_image):
        print(image)
        print(result_label)
        cv2.imwrite(examdirname + str(result_label) + "/" + str(i) + ".png", 255 * np.array(image))
        i += 1
