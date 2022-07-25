import cv2
import numpy as np
import math, imutils
import os, json
from math import copysign, log10

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4


# print('亮度:', cap.get(cv2.CAP_PROP_BRIGHTNESS))
# print('对比度:', cap.get(cv2.CAP_PROP_CONTRAST))
# print('饱和度:', cap.get(cv2.CAP_PROP_SATURATION))
# print('色调:', cap.get(cv2.CAP_PROP_HUE))
# print('曝光度:', cap.get(cv2.CAP_PROP_EXPOSURE))

def hu_moment(im):
    # im = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow('im', im)
    # cv2.waitKey(0)
    moments = cv2.moments(im)
    huMoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
        huMoments[i] = round(huMoments[i][0], 3)
    hu = []
    for item in huMoments:
        hu.append(float(item[0]))
    return hu


def hu_data_load():
    f = open('data.json')
    data = json.load(f)
    return data


def crop_img(img, x, y, w, h):
    crop_img = img[y:y + h, x:x + w]
    return crop_img


def compare_hu_moment(file, hu, hu_data):
    ran = 0.2
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


dir = 'match_data'
files = os.listdir(dir)
file_data = []
for file in files:
    template = cv2.imread(dir + '\\' + file)
    file_data.append({'template': template, 'file': file})

ret, cv_img = cap.read()
while True:
    # 擷取影像
    ret, img = cap.read()
    dir = 'match_data'
    files = os.listdir(dir)
    hu_data = hu_data_load()
    # print(hu_data)
    data = []
    # for file in files:
    for item in file_data:
        template = item['template']
        file = item['file']
        h, w, l = template.shape
        res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        max_val_ncc = '{:.3f}'.format(max_val)
        # print('{:.3f}'.format(max_val))
        if float(max_val_ncc) > 0.95:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            crop = crop_img(img,
                            top_left[0], top_left[1],
                            abs(bottom_right[0] - top_left[0]),
                            abs(bottom_right[0] - top_left[0]))
            cv2.imshow(str(file), crop)
            hu = hu_moment(crop)
            hu_check = False
            if str(file) in hu_data.keys():
                HU = compare_hu_moment(str(file), hu, hu_data)
                # print(HU)
                if HU is True:
                    hu_check = True
            data.append({'name': str(file), 'top_left': top_left, 'bottom_right': bottom_right, 'hu_check': hu_check})

    for item in data:
        cv2.rectangle(img, item['top_left'], item['bottom_right'], (0, 255, 0), 2)
        if item['hu_check'] == True:
            cv2.putText(img, item['name'], item['top_left'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('target', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
