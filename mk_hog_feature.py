import os
import cv2
import numpy as np
from skimage.feature import hog


class MakeFeature:
    def __init__(self, setting_yaml):
        hog_parameter = setting_yaml["hog_parameter"]
        self.hog_orientation_parameter = hog_parameter["orientation"]
        self.pixels_per_cell = hog_parameter["pixels_per_cell"]
        self.cells_per_block = hog_parameter["cells_per_block"]

    # 正規化用
    def normalize(self, image):
        max1 = np.max(image)
        min1 = np.min(image)
        if float(max1 - min1) == 0:
            return image
        else:
            y = (image - min1)*(255 / float(max1 - min1))
            return y.astype(np.int64)

    # hogの特徴量抽出用
    # image_df : pandas data frame
    # column : pandas data frameの画像が格納されている場所のカラム名
    def hog_detector(self, image_df, column, do_equalizeHist=False, do_normalize=False):
        for image in image_df[column]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            trans_image = gray_image.copy()
            # if option == 1:
            #     trans_image = cv2.equalizeHist(gray_image)
            # if option == 2:
            #     trans_image = self.normalize(gray_image)
            # if option == 3:
            #     equ = cv2.equalizeHist(gray_image)
            #     trans_image = self.normalize(equ)

            if do_equalizeHist is True:
                trans_image = cv2.equalizeHist(trans_image)

            if do_normalize is True:
                trans_image = self.normalize(trans_image)

            ppc = self.pixels_per_cell
            cpb = self.cells_per_block
            fd, hog_image = hog(
                trans_image,
                orientations=self.hog_orientation_parameter,
                pixels_per_cell=(ppc, ppc),
                cells_per_block=(cpb, cpb),
                visualise=True
            )
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), sharex=True, sharey=True)
        # cv2.imshow("hog_vector", hog_image)
        return fd

    def makearray(self, hog_array, label_type):
        # hog_array = np.loadtxt(csvfilepath, delimiter = ",")
        if label_type == 1:
            label_array = np.ones(len(hog_array))
        else:
            label_array = np.zeros(len(hog_array))
        return [hog_array, label_array]

if __name__ == "__main__":
    setting_dict = {
        'hog_parameter': {
            'orientation': 9,
            'pixels_per_cell': 6,
            'cells_per_block': 2
        }
    }
    hog_maker = MakeFeature(setting_yaml=setting_dict)
    input_image = cv2.imread(".//dataset/ped_images/img_04125.pgm")
    hog_feature = hog_maker.hog_detector(
        image=input_image,
        option=3,
        hog_feature=False
    )
    print(len(hog_feature))
