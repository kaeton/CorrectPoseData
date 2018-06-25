import cv2
import yaml
import numpy as np

class LabelingFrame:
    def __init__(self):
        self.frame_label = []
        self.train_frame = []

    def labeling(self, src):
        # '''
        # :param src:
        # npndarray:image data array
        #
        # :return:
        # frame label
        # '''
        label_name = 0
        prev_label = 0

        for frame in src:
            try:
                cv2.imshow(winname="frame_{label_name}", mat=frame)
            except:
                break

            input_key = cv2.waitKey()
            print(input_key)
            # skip function
            if input_key == ord('q'):
                return 0
            elif input_key == 27 or input_key == 13:
                self.frame_label.append(-1)
                continue
                # self.frame_label.append(prev_label)
                # self.train_frame.append(frame)
            elif input_key >= ord('0') and input_key <= ord('9'):
                label = input_key - ord('0')
                self.frame_label.append(label)
                self.train_frame.append(frame)
                prev_label = label

            label_name += 1

    def check_label(self):
        for i, frame in enumerate(self.train_frame):
            cv2.imshow(winname=label, mat=frame)
            input_key = cv2.waitKey()
            if input_key >= ord('0') and input_key <= ord('9'):
                self.frame_label[i] = input_key - ord('0')
            elif input_key == 27:
                continue

    def output_label(self, out_filename):
        print(out_filename)
        label = np.array(self.frame_label)
        print(label)
        np.savetxt(out_filename, label, delimiter=",")

if __name__ == "__main__":
    f = open("setting.yaml", "r+")
    setting = yaml.load(f)


    for label in setting["filename"]:
        for filepath in setting["filename"][label]:
            frame_feature = []
            src_video = cv2.VideoCapture(filepath)

            ret, frame = src_video.read()
            frame_feature.append(frame)
            while ret:
                ret, frame = src_video.read()
                frame_feature.append(frame)

            labelmaker = LabelingFrame()
            labelmaker.labeling(src=frame_feature)
            labelmaker.output_label(out_filename=filepath+".csv")
            # labelmaker.labeling()
