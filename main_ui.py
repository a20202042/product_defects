import numpy as np
import cv2, sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qt import Ui_Form
import math
from numpy import copysign, log10
import gvar, imutils, os, json
from skimage.metrics import structural_similarity


class VideoThread(QThread):
    change_pixmap_signal_cam = pyqtSignal(np.ndarray)
    change_pixmap_signal_get_object_img = pyqtSignal(np.ndarray)
    change_pixmap_signal_circle_img = pyqtSignal(np.ndarray)
    change_pixmap_signal_classfication_img = pyqtSignal(np.ndarray)
    change_pixmap_signal_find_text = pyqtSignal(np.ndarray)
    change_pixmap_signal_compare = pyqtSignal(np.ndarray)
    part_text = pyqtSignal([str], [str], [str])
    part_img = pyqtSignal([str])
    circle_text = pyqtSignal([list])
    signal_compare_text = pyqtSignal([str])

    # change_pixmap_signal_hand = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self._run_flag = True
        dir = 'match_data'
        files = os.listdir(dir)
        self.file_data = []
        for file in files:
            template = cv2.imread(dir + '\\' + file)
            self.file_data.append({'template': template, 'file': file})
        self.hu_data = self.load_hu_data()

    def load_hu_data(self):
        f = open('data.json')
        data = json.load(f)
        # print(data)
        return data

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
        cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
        cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
        cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
        cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4
        while self._run_flag:
            ret, cv_img = cap.read()
            cv_img = self.back_ground_remove(cv_img)  # 去背
            if gvar.start is True:
                ret, cv_img = cap.read()
                cv_img = self.back_ground_remove(cv_img)  # 去背
                part_name = str()
                object_img, rotated_img, rotated_img_2, circle_detect_img = cv_img.copy(), cv_img.copy(), cv_img.copy(), cv_img.copy()
                # cv2.imshow('cv', cv_img)
                # cv2.waitKey(0)
                self.change_pixmap_signal_cam.emit(cv_img) #設定影像參數並將畫面顯示於指定Label中
                box_img, box_compact_img, box = self.box_compact(cv_img) #最小矩形繪製
                # crop_img = self.crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]), object_img)
                # print(crop_img.shape[1])
                # img = self.resize_img(crop_img)
                angle = self.angle_calculate(box) #參數計算
                rotated_img = self.rotate_img(rotated_img, angle) #縮放成指定大小
                # cv2.imshow('cv', rotated_img)
                # cv2.waitKey(0)
                not_use, not_use_2, box = self.box_compact(rotated_img) #最小矩形繪製
                crop_img = self.rotate_img(rotated_img_2, angle)#縮放成指定大小
                compare_img = self.crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]),
                                        crop_img, 0)
                crop_img = self.crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]),
                                     crop_img, 3)
                text_img = crop_img.copy()

                # stust = cv2.imwrite('match_data/ASSY-14526561-1.png', crop_img)
                # stust = cv2.imwrite('match_data/14526862-1.png', crop_img)
                # crop_img = self.resize_img(crop_img)
                self.change_pixmap_signal_get_object_img.emit(crop_img)
                for item in self.file_data:
                    template = item['template']
                    file = item['file']
                    h, w, l = template.shape
                    # print(h, w, l)
                    if crop_img.shape[0] < h and crop_img.shape[1] < w:
                        res = cv2.matchTemplate(crop_img, template, cv2.TM_CCORR_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        max_val_ncc = '{:.3f}'.format(max_val)
                        # print(max_val_ncc)
                        if float(max_val_ncc) > 0.9:
                            match = True
                            top_left = max_loc
                            bottom_right = (top_left[0] + w, top_left[1] + h)
                            hu = self.hu_moment(crop_img)
                            if str(file) in self.hu_data.keys():
                                HU = self.compare_hu_moment(str(file), hu, self.hu_data)
                                if HU is True:
                                    # print('in_data')
                                    self.part_text.emit(file)
                                    self.part_img.emit(file)
                                    part_name = file
                                    check = True
                                    text = file
                                    put_img = crop_img.copy()
                                    cv2.putText(put_img, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (0, 0, 255), 2, cv2.LINE_AA)
                                    self.change_pixmap_signal_classfication_img.emit(put_img)
                                else:
                                    check = True
                            else:
                                check = False
                                self.part_text.emit('未知零件')
                        else:
                            check = False
                            self.part_text.emit('未知零件')
                if check is True:
                    circle_detect_img = crop_img.copy()
                    circle_detect_img, circles_data = self.circle_detect(circle_detect_img)
                    self.change_pixmap_signal_circle_img.emit(circle_detect_img)
                    if circles_data != []:
                        self.circle_text.emit(circles_data)
                # 文字位置
                text_img = self.rotate_img(text_img, -90)
                text_img = self.text_find(text_img)
                self.change_pixmap_signal_find_text.emit(text_img)
                # --------
                # 比較
                compare, compare_check = self.compare_find(part_name, compare_img)
                self.change_pixmap_signal_compare.emit(compare)
                if compare_check is True:
                    self.signal_compare_text.emit('檢測到瑕疵')
                elif compare_check is False:
                    self.signal_compare_text.emit('無瑕疵')
                # --------
                gvar.start = False
            else:
                self.change_pixmap_signal_cam.emit(cv_img)
        cap.release()

    def compare_find(self, or_img, check_img):
        before = cv2.imread('match_data\\' + or_img)
        (h, w, d) = before.shape
        before = before[20:h - 20, 20:w - 20]
        before = cv2.resize(before, (w, h), interpolation=cv2.INTER_AREA)
        # after = cv2.imread('14526862-1_compare.png')
        after = cv2.resize(check_img, (w, h), interpolation=cv2.INTER_AREA)
        # range_data = self.find_text_position(after)
        # print(range_data)
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(before_gray, after_gray, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))
        diff = (diff * 255).astype("uint8")
        diff_box = cv2.merge([diff, diff, diff])
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(before.shape, dtype='uint8')
        filled_after = after.copy()
        compare = False
        for c in contours:
            area = cv2.contourArea(c)
            if area > 3500:
                x, y, w, h = cv2.boundingRect(c)
                # print(x, y, w, h)
                # print(range_data)
                # if range_data[0] < x < range_data[2] or range_data[0] < x + w < range_data[2]:
                #     pass
                # elif x < range_data[0] < x + w or x < range_data[2] < x + w:
                #     pass
                # else:
                cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 5)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 5)
                cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 5)
                cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
                compare = True
        return after, compare

    def text_find(self, img):
        # 讀取圖片
        # # imagePath = '14526862-1.png'
        # imagePath = 'ASSY-14526561-1.png'
        # img = cv2.imread(imagePath)
        # 轉化成灰度圖
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 利用Sobel邊緣檢測生成二值圖
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        # 二值化
        ret, binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # 膨脹、腐蝕
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 9))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
        # 膨脹一次，讓輪廓突出
        dilation = cv2.dilate(binary, element2, iterations=1)
        # 腐蝕一次，去掉細節
        erosion = cv2.erode(dilation, element1, iterations=1)
        # 再次膨脹，讓輪廓明顯一些
        dilation2 = cv2.dilate(erosion, element2, iterations=2)
        #  查詢輪廓和篩選文字區域
        region = []
        contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            # 計算輪廓面積，並篩選掉面積小的
            area = cv2.contourArea(cnt)
            if (area < 5000):
                continue
            # 找到最小的矩形
            rect = cv2.minAreaRect(cnt)
            # box是四個點的座標
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 計算高和寬
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])
            # 根據文字特徵，篩選那些太細的矩形，留下扁的
            if (height > width * 1.2):
                continue
            region.append(box)
        copy_img = img.copy()
        # 繪製輪廓
        for box in region:
            cv2.drawContours(copy_img, [box], 0, (255, 255, 0), 5)
            c = cv2.boundingRect(box)
        return copy_img

    def circle_detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉灰階

        gray = cv2.medianBlur(gray, 9)  # 使用中值平滑對影圖像進行模糊處理

        rows = gray.shape[0]

        circles = cv2.HoughCircles(gray,
                                   cv2.HOUGH_GRADIENT,
                                   minDist=40,
                                   # 圓心距離
                                   dp=1.2,
                                   # 檢測圓心的累加器精度和圖像精度比的倒數(1=相同分辨綠，2=累加器是輸入圖案一半大的寬高)
                                   param1=150,
                                   # canny檢測的高闊值，低闊值為一半
                                   param2=50,
                                   # 圓心的累加器闊值，越小檢測更多的圓，越大越精確
                                   minRadius=1,
                                   # 最小半徑
                                   maxRadius=40)
        # 最大半徑
        circles_data = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 四捨五入(np.around)後，轉換成16位無符號整數(np.uint16)。
            number = 1
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(img, center, 1, (0, 100, 100), 5)
                # (原圖、圓心、半徑、顏色、粗細)
                # circle outline
                radius = i[2]
                cv2.circle(img, center, radius, (255, 255, 255), 5)
                circles_data.append({'number': number, 'center': center, 'r': i})  # 每個圈的數據
                number = number + 1  # 計數器加1
        # print(circles_data)
        for item in circles_data:
            text = '%s' % item['number']
            cv2.putText(img, text, item['center'], cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 5, cv2.LINE_AA)
            # 繪製文字(圖片影像/繪製的文字/左上角坐標/字體/字體大小/顏色/字體粗細/字體線條種類)
        # cv.imshow("detected circles", src)
        #
        # cv.waitKey(0)
        return img, circles_data

    def compare_hu_moment(self, file, hu, hu_data):
        ran = 0.15
        hu1, hu2, hu3 = bool, bool, bool
        H_U = False
        # print(hu_data[str(file)]['hu1'][0] * (1 - ran), ':%s:' % hu[0], hu_data[str(file)]['hu1'][1] * (1 + ran))
        # print(hu_data[str(file)]['hu2'][0] * (1 - ran), ':%s:' % hu[1], hu_data[str(file)]['hu2'][1] * (1 + ran))
        # print(hu_data[str(file)]['hu3'][0] * (1 - ran), ':%s:' % hu[2], hu_data[str(file)]['hu3'][1] * (1 + ran))
        if hu_data[str(file)]['hu1'][0] * (1 - ran) < hu[0] < hu_data[str(file)]['hu1'][1] * (1 + ran):
            hu1 = True
        if hu_data[str(file)]['hu2'][0] * (1 - ran) < hu[1] < hu_data[str(file)]['hu2'][1] * (1 + ran):
            hu2 = True
        if hu_data[str(file)]['hu3'][0] * (1 - ran) < hu[2] < hu_data[str(file)]['hu3'][1] * (1 + ran):
            hu3 = True
        if hu1 is True and hu2 is True and hu3 is True:
            H_U = True
        return H_U

    def hu_moment(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, im = cv2.threshold(im, 60, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(im)
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):
            huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
            huMoments[i] = round(huMoments[i][0], 3)
        hu = []
        for item in huMoments:
            hu.append(float(item[0]))
        return hu

    def rotate_img(self, img, angle):
        rotated_img = imutils.rotate_bound(img, angle) #縮小成適當大小
        # rotated_img
        return rotated_img

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
        # range = 1
        x = min([left_point_x, right_point_x]) - range
        w = abs(right_point_x - left_point_x) + range * 2
        y = min([top_point_y, bottom_point_y]) - range
        h = abs(top_point_y - bottom_point_y) + range * 2
        crop_img = crop_img[y:y + h, x:x + w]
        return crop_img

    def angle_calculate(self, box):
        # print(point_1, point_2)
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
        # print(top_point, bottom_point, right_point, left_point)
        point_data = [top_point, bottom_point, right_point, left_point]
        # angle_ = angle(right_point, left_point)
        angle_ = math.atan2(bottom_point[0] - right_point[0], bottom_point[1] - right_point[1]) / math.pi * 180
        if angle_ > -45:
            pass
        elif angle_ < -45:
            angle_ = angle_ + 90
        return angle_


class App(QWidget, Ui_Form):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.thread = VideoThread()
        self.disply_width, self.display_height = 640, 360
        self.thread.start()
        self.thread.change_pixmap_signal_cam.connect(self.update_image_cam)
        self.thread.change_pixmap_signal_get_object_img.connect(self.update_image_get_object_img)
        self.thread.change_pixmap_signal_circle_img.connect(self.update_image_circle_img)
        self.thread.change_pixmap_signal_classfication_img.connect(self.classfication_img)
        self.thread.change_pixmap_signal_find_text.connect(self.find_text)
        self.thread.change_pixmap_signal_compare.connect(self.compare)
        self.thread.part_text.connect(self.part_text)
        self.thread.part_img.connect(self.part_img)
        self.thread.circle_text.connect(self.part_circle_text)
        self.thread.signal_compare_text.connect(self.compare_text)
        self.ui.pushButton.clicked.connect(self.start)
        self.ui.pushButton_2.clicked.connect(self.stop)
        BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(BASE_DIR + '\\ico.ico'))

    def start(self):
        gvar.start = True

    def stop(self):
        gvar.start = False

    def part_img(self, file_name):
        file = 'match_data' + '\\' + file_name
        img = cv2.imread(file)
        img = self.resize_img(img)
        w, h, l = img.shape
        qt_img = self.convert_cv_qt(img, w, h)
        self.ui.part_data_img.setPixmap(qt_img)

    def part_text(self, part_name):
        part_name = part_name.split('.')[0]
        self.ui.label_13.setText(str(part_name))

    def part_circle_text(self, circle_data):
        print(circle_data)
        # text = str()
        # for item in circle_data:
        text = '孔數量：'
        number = circle_data[-1]['number']
        text = text + str(number)
        # for item in circle_data:
        #     text = '' + text + ''
        self.ui.label_10.setText(str(text))

    def compare_text(self, text):
        self.ui.label_12.setText(text)

    def convert_cv_qt(self, cv_img, im_w, im_h):  # 輸入label
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # 轉成RGB
        h, w, ch = rgb_image.shape  # 取得參數，（高度、寬度、通道數）
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line,
                                            QtGui.QImage.Format_RGB888)  # 讀取圖片顯示在QLabel上
        p = convert_to_Qt_format.scaled(im_w, im_h, Qt.KeepAspectRatio)  #固定長寬比
        return QPixmap.fromImage(p) #格式轉換Pixmap>Image

    def resize_img(self, img):
        (w, h, l) = img.shape
        if int(w) > 180:
            scale = 2.0
            dim = (int(h / scale), int(w / scale))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    @pyqtSlot(np.ndarray)
    def update_image_cam(self, cv_img):
        w, h, l = cv_img.shape  # 圖像參數（高度、寬度、通道數）
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.camera.setPixmap(qt_img)  #顯示於Label中

    def update_image_get_object_img(self, cv_img):
        cv_img = self.resize_img(cv_img)
        w, h, l = cv_img.shape
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.get_object_img.setPixmap(qt_img)

    def classfication_img(self, cv_img):
        cv_img = self.resize_img(cv_img)
        w, h, l = cv_img.shape
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.part_classification.setPixmap(qt_img)

    def compare(self, cv_img):
        cv_img = self.resize_img(cv_img)
        w, h, l = cv_img.shape
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.flaw.setPixmap(qt_img)

    def find_text(self, cv_img):
        cv_img = self.resize_img(cv_img)
        w, h, l = cv_img.shape
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.label_word_recognition.setPixmap(qt_img)

    def update_image_circle_img(self, cv_img):
        cv_img = self.resize_img(cv_img)
        w, h, l = cv_img.shape
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.hole_part.setPixmap(qt_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
