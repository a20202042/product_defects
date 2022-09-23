import cv2, sys
import os
import numpy as np
from numpy import copysign, log10
import json
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qt_1 import Ui_Form
import math
import gvar, imutils


class VideoThread(QThread):
    change_pixmap_signal_cam = pyqtSignal(np.ndarray)
    change_pixmap_signal_cut_img = pyqtSignal(np.ndarray)
    return_crop_img = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def stop(self):
        self._run_flag = False

    def run(self):
        cap = cv2.VideoCapture()
        # The device number might be 0 or 1 depending on the device and the webcam
        cap.open(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        # while (True):
        #     ret, frame = cap.read()
        #     cv2.imshow('frame', frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
        # cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
        # cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
        # cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
        # cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4
        while self._run_flag:
            ret, cv_img = cap.read()
            self.change_pixmap_signal_cam.emit(cv_img)
            if gvar.start is True:
                ret, cv_img = cap.read()
                cv_img = self.back_ground_remove(cv_img)  # 去背
                part_name = str()
                object_img, rotated_img, rotated_img_2, circle_detect_img = cv_img.copy(), cv_img.copy(), cv_img.copy(), cv_img.copy()
                try:
                    box_img, box_compact_img, box = self.box_compact(cv_img)  # 最小矩形繪製
                except:
                    self.change_pixmap_signal_cut_img.emit(cv2.imread('error.jpg'))  # 設定影像參數並將畫面顯示於指定Label中
                else:
                    angle = self.angle_calculate(box)  # 參數計算
                    rotated_img = self.rotate_img(rotated_img, angle)  # 縮放成指定大小
                    not_use, not_use_2, box = self.box_compact(rotated_img)  # 最小矩形繪製
                    crop_img = self.rotate_img(rotated_img_2, angle)  # 縮放成指定大小
                    compare_img = self.crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]),
                                            crop_img, 0)
                    crop_img = self.crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]),
                                         crop_img, 3)
                    text_img = crop_img.copy()

                    self.change_pixmap_signal_cut_img.emit(compare_img)  # 設定影像參數並將畫面顯示於指定Label中

                    if gvar.crop_check is True:
                        self.return_crop_img.emit(compare_img)
                        gvar.crop_check = False

        cap.release()

    def back_ground_remove(self, img):
        low_green = np.array([0, 0, 90])  # 創建矩陣
        high_green = np.array([255, 255, 255])  # 創建矩陣
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 轉hsv
        mask = cv2.inRange(imgHSV, low_green, high_green)  # 某顏色區域影像位置，(原圖,低於low_green和高於high_green，圖像值變為0)
        res = cv2.bitwise_and(img, img, mask=mask)  # 遮罩印在原來的圖片上(挖掉圖像)
        return res

    def box_compact(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 圖片二值化，(img, 閥值, 最大灰度值, 使用的二值化方法)
        kernel = np.ones((3, 3), dtype=np.uint8)
        # 設定矩陣大小和類型，(shape(個數,寬,高),dtype,order,like)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 形態學變化函數，(img,op,kernal(濾波器))，cv2.MORPH_CLOSE閉運算。
        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # 尋找輪廓，cv2.RETR_TREE檢索完整層次結構中的輪廓，cv2.CHAIN_APPROX_NONE存储所有的輪廓點，相鄰的兩個點的像素位置差不超過1
        cnt = max(contours, key=cv2.contourArea)  # 計算面積
        x, y, w, h = cv2.boundingRect(cnt)  # cv2.findContours所取得x, y, w, h ，計算出包覆輪廓的最小的正矩形
        img_2, img_3 = img.copy(), img.copy()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
        # 繪製矩形
        # cv2.imshow('img', img)
        rect = cv2.minAreaRect(cnt)  # 包護輪廓的最小斜矩形，中心點座標,寬高,旋轉角度
        box = cv2.boxPoints(rect)  # 獲得矩形4個頂點
        box = np.int0(box)  # 取整
        cv2.drawContours(img_2, [box], 0, (0, 0, 255), 2)
        # 繪製出輪廓，(圖像,輪廓參數,繪製輪廓數,顏色,寬度)
        return img, img_2, box

    def crop(self, left_point_x, right_point_x, top_point_y, bottom_point_y, crop_img, range):
        x = min([left_point_x, right_point_x]) - range
        w = abs(right_point_x - left_point_x) + range * 2
        y = min([top_point_y, bottom_point_y]) - range
        h = abs(top_point_y - bottom_point_y) + range * 2
        crop_img = crop_img[y:y + h, x:x + w]
        return crop_img

    def angle_calculate(self, box):
        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])
        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
        # print(left_point_y, right_point_y, top_point_x, bottom_point_x)
        top_point = [top_point_x, top_point_y]
        bottom_point = [bottom_point_x, bottom_point_y]
        right_point = [right_point_x, right_point_y]
        left_point = [left_point_x, left_point_y]
        point_data = [top_point, bottom_point, right_point, left_point]
        angle_ = math.atan2(bottom_point[0] - right_point[0], bottom_point[1] - right_point[1]) / math.pi * 180
        if angle_ > -45:
            pass
        elif angle_ < -45:
            angle_ = angle_ + 90
        return angle_

    def rotate_img(self, img, angle):
        rotated_img = imutils.rotate_bound(img, angle)  # 縮小成適當大小
        return rotated_img


class App(QWidget, Ui_Form):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.disply_width, self.display_height = 640, 360
        self.thread = VideoThread()
        self.thread.start()
        self.thread.change_pixmap_signal_cam.connect(self.update_image_cam)
        self.ui.RUN.clicked.connect(self.start)
        self.ui.cut.clicked.connect(self.cut)
        self.ui.hu.clicked.connect(self.hu)
        self.ui.Stop.clicked.connect(self.stop)
        self.ui.toolButton.clicked.connect(self.path)
        self.thread.change_pixmap_signal_cut_img.connect(self.cut_down)
        self.thread.return_crop_img.connect(self.save_crop_img)

    def start(self):
        gvar.start = True

    def stop(self):
        gvar.start = False
        self.ui.label.clear()

    def save_crop_img(self, cv_img):
        path = self.ui.line_path.text()
        name = self.ui.line_name.text()
        path = path.split('/')
        sep = '\\'
        path_name = sep.join(path) + '\\' + name + '.png'
        print(path_name)
        cv2.imwrite(path_name, cv_img)

    def cut(self):
        gvar.crop_check = True

    def hu(self):
        gvar.hu_json = True
        path = self.ui.line_path.text()
        path = path.split('/')
        sep = '\\'
        path_name = sep.join(path)
        self.hu_json(path_name)

    def hu_json(self, path_name):
        all_data = {}
        files = os.listdir(path_name)
        for file in files:
            print(file)
            all_hu = []
            for i in range(0, 180):
                im = cv2.imread(path_name + '\\' + file, cv2.IMREAD_GRAYSCALE)
                im = self.rotate_img_hu(im, i)
                hu = self.hu_moment(im)
                all_hu.append(hu)
                print(hu)
            hu_1 = []
            hu_2 = []
            hu_3 = []
            for item in all_hu:
                hu_1.append(item[0])
                hu_2.append(item[1])
                hu_3.append(item[2])

            print('hu1:' + str(min(hu_1)) + ' ' + str(max(hu_1)))
            print('hu2:' + str(min(hu_2)) + ' ' + str(max(hu_2)))
            print('hu3:' + str(min(hu_3)) + ' ' + str(max(hu_3)))

            item_hu = {'hu1': [min(hu_1), max(hu_1)],
                       'hu2': [min(hu_2), max(hu_2)],
                       'hu3': [min(hu_3), max(hu_3)]}
            all_data.update({str(file): item_hu})

        with open('data.json', 'w') as f:
            json.dump(all_data, f)

        f = open('data.json')
        data = json.load(f)
        print(data)

    def rotate_img_hu(self, img, angle):
        (h, w) = img.shape  # 讀取圖片大小
        center = (w // 2, h // 2)  # 找到圖片中心

        # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 第三個參數變化後的圖片大小
        rotate_img = cv2.warpAffine(img, M, (w, h))

        return rotate_img

    def hu_moment(self, im):
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

    def path(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Openfolder", "./ ")
        self.ui.line_path.setText(folder_path)

    def convert_cv_qt(self, cv_img, im_w, im_h):  # 輸入label
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # 轉成RGB
        except:
            rgb_image = np.zeros((1280, 720, 3), np.uint8)
        h, w, ch = rgb_image.shape  # 取得參數，（高度、寬度、通道數）
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line,
                                            QtGui.QImage.Format_RGB888)  # 讀取圖片顯示在QLabel上
        p = convert_to_Qt_format.scaled(im_w, im_h, Qt.KeepAspectRatio)  # 固定長寬比
        return QPixmap.fromImage(p)  # 格式轉換Pixmap>Image

    @pyqtSlot(np.ndarray)
    def update_image_cam(self, cv_img):
        w, h, l = cv_img.shape  # 圖像參數（高度、寬度、通道數）
        w = 9 * w / 10
        h = 9 * h / 10
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.picture.setPixmap(qt_img)  # 顯示於Label中

    @pyqtSlot(np.ndarray)
    def cut_down(self, cv_img):
        w, h, l = cv_img.shape  # 圖像參數（高度、寬度、通道數）
        # if w >= 337 and h > 356:
        #     w = 6 * w / 10
        #     h = 6 * h / 10

        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.label.setPixmap(qt_img)  # 顯示於Label中


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
