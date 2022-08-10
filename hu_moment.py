import os

import cv2, sys
from numpy import copysign, log10
import json
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qt_1 import Ui_Form



def hu_moment(im):
    # im = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im, 60, 255, cv2.THRESH_BINARY)
    # cv2.imshow('im', im)
    moments = cv2.moments(im)
    huMoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
        huMoments[i] = round(huMoments[i][0], 3)
    hu = []
    for item in huMoments:
        hu.append(float(item[0]))
    return hu


def rotate_img(img, angle):
    (h, w) = img.shape  # 讀取圖片大小
    center = (w // 2, h // 2)  # 找到圖片中心

    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h))

    return rotate_img


# all_data = {}
# files = os.listdir('match_data')
# for file in files:
#     # img = cv2.imread('match_data\\' + file)
#     # img, circles_data = circle_detect(img)
#     # cv2.imshow('test', img)
#     # cv2.waitKey(0)
#     print(file)
#     all_hu = []
#     for i in range(0, 180):
#         im = cv2.imread('match_data\\' + file, cv2.IMREAD_GRAYSCALE)
#         im = rotate_img(im, i)
#         hu = hu_moment(im)
#         all_hu.append(hu)
#         # print(hu)
#     hu_1 = []
#     hu_2 = []
#     hu_3 = []
#     for item in all_hu:
#         hu_1.append(item[0])
#         hu_2.append(item[1])
#         hu_3.append(item[2])
#
#     print('hu1:' + str(min(hu_1)) + ' ' + str(max(hu_1)))
#     print('hu2:' + str(min(hu_2)) + ' ' + str(max(hu_2)))
#     print('hu3:' + str(min(hu_3)) + ' ' + str(max(hu_3)))
#
#     item_hu = {'hu1': [min(hu_1), max(hu_1)],
#                'hu2': [min(hu_2), max(hu_2)],
#                'hu3': [min(hu_3), max(hu_3)]}
#     all_data.update({str(file): item_hu})
#
# with open('data.json', 'w') as f:
#     json.dump(all_data, f)
#
# f = open('data.json')
# data = json.load(f)
# print(data)
class VideoThread(QThread):
    change_pixmap_signal_cam = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self._run_flag = True
    def run(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
        cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
        cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
        cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
        cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4
        while self._run_flag:
            ret, cv_img = cap.read()
            self.change_pixmap_signal_cam.emit(cv_img)



class App(QWidget, Ui_Form):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.disply_width, self.display_height = 640, 360
        self.thread = VideoThread()
        self.thread.start()
        self.thread.change_pixmap_signal_cam.connect(self.update_image_cam)


    def convert_cv_qt(self, cv_img, im_w, im_h):  # 輸入label
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line,
                                            QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(im_w, im_h, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(np.ndarray)
    def update_image_cam(self, cv_img):
        w, h, l = cv_img.shape
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.picture.setPixmap(qt_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
