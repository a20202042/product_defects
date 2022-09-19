import cv2
from PIL import Image
import pytesseract
import plot
import np
import os, sys
import csv
import time

# 匯入圖檔
img_input = 'text_recognition_photo\\test20_1.jpeg'  # Input圖片檔名
img = cv2.imread(img_input)
cv2.imshow('img_input', img)

# 銳化
blur_img = cv2.GaussianBlur(img, (0, 0), 30)
usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
cv2.imshow('blur_img', blur_img)
cv2.imshow('usm', usm)

# 二次銳化
blur_img_2 = cv2.GaussianBlur(img, (0, 0), 80)
usm_2 = cv2.addWeighted(usm, 1.5, blur_img_2, -0.5, 0)
cv2.imshow('blur_img_2', blur_img_2)
cv2.imshow('usm_2', usm_2)

# 調整對比及亮度
alpha = 1.7 #對比度
beta = 0.4 #亮度
adjusted = cv2.convertScaleAbs(usm_2, alpha = alpha, beta = beta)
cv2.imshow('abjusted', adjusted)

# 轉灰階
grayscale = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale', grayscale)

# 膨脹和腐蝕，去除黑白點
kernel = np.ones((1, 1), np.uint8)
swell = cv2.dilate(grayscale, kernel, iterations = 4) #膨脹
corrosion = cv2.erode(swell, kernel, iterations = 1) #腐蝕
swell_2 = cv2.dilate(corrosion, kernel, iterations = 4) #二次膨脹
cv2.imshow('swell', swell)
cv2.imshow('corrosion', corrosion)
cv2.imshow('swell_2', swell_2)

# 圖片二值化
ret, binary = cv2.threshold(swell_2, 13, 255, cv2.THRESH_BINARY) #把圖片二值化，()內參數分別是：轉檔圖片名稱、
cv2.imshow('binary', binary)

# 改變圖片大小
crop_x, crop_y = binary.shape
# print(crop_x,'\n' , crop_y)
magnification_x = 1.5
magnification_y = 1.5
enlarge = cv2.resize(binary, (int(crop_y * magnification_y), int(crop_x * magnification_x)))
cv2.imshow('enlarge', enlarge)

# 膨脹和腐蝕，去除黑白點
# swell_3 = cv2.dilate(enlarge, kernel, iterations = 1) #膨脹
# cv2.imshow('swell_3', swell_3)

# 辨識圖片文字
text = pytesseract.image_to_string(enlarge, lang='eng')  # 辨識圖片中文字部分
# text_data = pytesseract.image_to_data(enlarge, lang='eng')  # 辨識圖片中文字部分
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) #紀錄現在的時間
if text == '':
    print('沒有辨識出結果\n')
    # print(text_data)
    cv2.imwrite("text_recognition_photo\\test\\fail\\camara_fail_" + now + ".png", enlarge)  # 儲存匯入模組的圖片
else:
    print(text)
    # print('\n' + text_data)
    cv2.imwrite("text_recognition_photo\\test\\success\\camara_success" + now + ".png", enlarge)  # 儲存匯入模組的圖片
cv2.waitKey(0)

# ----------------------------迴圈測試

# for abc in range(0, 250):
#     # 匯入圖檔
#     img_input = 'text_recognition_photo\\test20_1.jpeg'  # Input圖片檔名
#     img = cv2.imread(img_input)
#     # cv2.imshow('img_input', img)
#
#     # 銳化
#     blur_img = cv2.GaussianBlur(img, (0, 0), 30)
#     usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
#     # cv2.imshow('blur_img', blur_img)
#     # cv2.imshow('usm', usm)
#
#     # 二次銳化
#     blur_img_2 = cv2.GaussianBlur(img, (0, 0), 80)
#     usm_2 = cv2.addWeighted(usm, 1.5, blur_img_2, -0.5, 0)
#     # cv2.imshow('blur_img_2', blur_img_2)
#     # cv2.imshow('usm_2', usm_2)
#
#     # 調整對比及亮度
#     alpha = 1.7  # 對比度
#     beta = 0.4  # 亮度
#     adjusted = cv2.convertScaleAbs(usm_2, alpha=alpha, beta=beta)
#     # cv2.imshow('abjusted', adjusted)
#
#     # 轉灰階
#     grayscale = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('grayscale', grayscale)
#
#     # 膨脹和腐蝕，去除黑白點
#     kernel = np.ones((2, 2), np.uint8)
#     swell = cv2.dilate(grayscale, kernel, iterations=1)  # 膨脹
#     corrosion = cv2.erode(swell, kernel, iterations=2)  # 腐蝕
#     swell_2 = cv2.dilate(corrosion, kernel, iterations=2)  # 二次膨脹
#     # cv2.imshow('swell', swell)
#     # cv2.imshow('corrosion', corrosion)
#     # cv2.imshow('swell_2', swell_2)
#
#     # 圖片二值化
#     ret, binary = cv2.threshold(swell_2, abc, 255, cv2.THRESH_BINARY)  # 把圖片二值化，()內參數分別是：轉檔圖片名稱、
#     # cv2.imshow('binary', binary)
#
#     # 改變圖片大小
#     crop_x, crop_y = binary.shape
#     # print(crop_x,'\n' , crop_y)
#     magnification_x = 1.5
#     magnification_y = 1.5
#     enlarge = cv2.resize(binary, (int(crop_y * magnification_y), int(crop_x * magnification_x)))
#     # cv2.imshow('enlarge', enlarge)
#
#     # 膨脹和腐蝕，去除黑白點
#     # swell_3 = cv2.dilate(enlarge, kernel, iterations = 1) #膨脹
#     # cv2.imshow('swell_3', swell_3)
#
#     # 辨識圖片文字
#     text = pytesseract.image_to_string(enlarge, lang='eng')  # 辨識圖片中文字部分
#     # text_data = pytesseract.image_to_data(enlarge, lang='eng')  # 辨識圖片中文字部分
#     # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))  # 紀錄現在的時間
#     if text == '':
#         pass
#         # print('二值化的值：%s\n' %abc)
#         # print('沒有辨識出結果\n')
#         # print(text_data)
#         # cv2.imwrite("text_recognition_photo\\test\\fail\\camara_fail_" + now + ".png", enlarge)  # 儲存匯入模組的圖片
#     else:
#         print('二值化的值：%s' %abc + '***\n')
#         print(text + '\n')
#         # cv2.imwrite("text_recognition_photo\\test\\success\\camara_success" + now + ".png", enlarge)  # 儲存匯入模組的圖片
#     cv2.waitKey(0)

# ----------------------------手機測試
# # 匯入圖檔
# img_input = 'text_recognition_photo\\test16_2.jpg'  # Input圖片檔名
# img = cv2.imread(img_input)
# cv2.imshow('img_input', img)
#
# # 銳化
# blur_img = cv2.GaussianBlur(img, (0, 0), 50)
# cv2.imshow('blur_img', blur_img)
# usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
# cv2.imshow('usm', usm)
#
# # 調整對比及亮度
# alpha = 1.7 #對比度
# beta = 0 #亮度
# adjusted = cv2.convertScaleAbs(usm, alpha = alpha, beta = beta)
# cv2.imshow('abjusted', adjusted)
#
# # 轉灰階
# grayscale = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayscale', grayscale)
#
# # 膨脹和腐蝕，去除黑白點
# kernel = np.ones((3, 3), np.uint8)
# swell = cv2.dilate(grayscale, kernel, iterations = 2) #膨脹
# corrosion = cv2.erode(swell, kernel, iterations = 2) #腐蝕
# swell_2 = cv2.dilate(corrosion, kernel, iterations = 1) #二次膨脹
# cv2.imshow('swell', swell)
# cv2.imshow('corrosion', corrosion)
# cv2.imshow('swell_2', swell_2)
#
# # # 圖片二值化
# ret, binary = cv2.threshold(corrosion, 150, 255, cv2.THRESH_BINARY) #把圖片二值化，()內參數分別是：轉檔圖片名稱、
# cv2.imshow('binary', binary)
#
# # 改變圖片大小
# crop_x, crop_y = binary.shape
# magnification_x = 0.4 #x方向的縮放倍數
# magnification_y = 0.4 #y方向的縮放倍數
# enlarge = cv2.resize(binary, (int(crop_y * magnification_y), int(crop_x * magnification_x)))
# cv2.imshow('enlarge', enlarge)
#
# # 儲存匯入模組辨識的圖片
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) #紀錄現在的時間
# print('現在的時間：%s' %now)
# cv2.imwrite("text_recognition_photo\\test\\phone_" + now + ".png", enlarge) #儲存匯入模組的圖片
#
# # 辨識圖片文字
# text = pytesseract.image_to_string(enlarge, lang='eng')  # 辨識圖片中文字部分
# text_data = pytesseract.image_to_data(enlarge, lang='eng')  # 辨識圖片中文字部分
# if text == '':
#     print('沒有辨識出結果\n')
#     print(text_data)
# else:
#     print(text)
#     print('\n' + text_data)
# cv2.waitKey(0)

# ----------------------------辨識成功案例（test20_1）----------------------------
# # 匯入圖檔
# img_input = 'text_recognition_photo\\test20_1.jpeg'  # Input圖片檔名
# img = cv2.imread(img_input)
# cv2.imshow('img_input', img)
#
# # 銳化
# blur_img = cv2.GaussianBlur(img, (0, 0), 30)
# usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
# cv2.imshow('blur_img', blur_img)
# cv2.imshow('usm', usm)
#
# # 二次銳化
# blur_img_2 = cv2.GaussianBlur(img, (0, 0), 80)
# usm_2 = cv2.addWeighted(usm, 1.5, blur_img_2, -0.5, 0)
# cv2.imshow('blur_img_2', blur_img_2)
# cv2.imshow('usm_2', usm_2)
#
# # 調整對比及亮度
# alpha = 1.7 #對比度
# beta = 0.4 #亮度
# adjusted = cv2.convertScaleAbs(usm_2, alpha = alpha, beta = beta)
# cv2.imshow('abjusted', adjusted)
#
# # 轉灰階
# grayscale = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayscale', grayscale)
#
# # 膨脹和腐蝕，去除黑白點
# kernel = np.ones((1, 1), np.uint8)
# swell = cv2.dilate(grayscale, kernel, iterations = 4) #膨脹
# corrosion = cv2.erode(swell, kernel, iterations = 1) #腐蝕
# swell_2 = cv2.dilate(corrosion, kernel, iterations = 4) #二次膨脹
# cv2.imshow('swell', swell)
# cv2.imshow('corrosion', corrosion)
# cv2.imshow('swell_2', swell_2)
#
# # 圖片二值化
# ret, binary = cv2.threshold(swell_2, 120, 255, cv2.THRESH_BINARY) #把圖片二值化，()內參數分別是：轉檔圖片名稱、
# cv2.imshow('binary', binary)
#
# # 改變圖片大小
# crop_x, crop_y = binary.shape
# # print(crop_x,'\n' , crop_y)
# magnification_x = 1.5
# magnification_y = 1.5
# enlarge = cv2.resize(binary, (int(crop_y * magnification_y), int(crop_x * magnification_x)))
# cv2.imshow('enlarge', enlarge)
#
# # 膨脹和腐蝕，去除黑白點
# # swell_3 = cv2.dilate(enlarge, kernel, iterations = 1) #膨脹
# # cv2.imshow('swell_3', swell_3)
#
# # 辨識圖片文字
# text = pytesseract.image_to_string(enlarge, lang='eng')  # 辨識圖片中文字部分
# text_data = pytesseract.image_to_data(enlarge, lang='eng')  # 辨識圖片中文字部分
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) #紀錄現在的時間
# if text == '':
#     print('沒有辨識出結果\n')
#     print(text_data)
#     cv2.imwrite("text_recognition_photo\\test\\fail\\camara_fail_" + now + ".png", enlarge)  # 儲存匯入模組的圖片
# else:
#     print(text)
#     print('\n' + text_data)
#     cv2.imwrite("text_recognition_photo\\test\\success\\camara_success" + now + ".png", enlarge)  # 儲存匯入模組的圖片
# cv2.waitKey(0)

# ----------------------------辨識成功案例（test20_2）----------------------------
# # 匯入圖檔
# img_input = 'text_recognition_photo\\test20_2.jpeg'  # Input圖片檔名
# img = cv2.imread(img_input)
# cv2.imshow('img_input', img)
#
# # 銳化
# blur_img = cv2.GaussianBlur(img, (0, 0), 30)
# usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
# cv2.imshow('blur_img', blur_img)
# cv2.imshow('usm', usm)
#
# # 二次銳化
# blur_img_2 = cv2.GaussianBlur(img, (0, 0), 80)
# usm_2 = cv2.addWeighted(usm, 1.5, blur_img_2, -0.5, 0)
# cv2.imshow('blur_img_2', blur_img_2)
# cv2.imshow('usm_2', usm_2)
#
# # 調整對比及亮度
# alpha = 1.7 #對比度
# beta = 0.4 #亮度
# adjusted = cv2.convertScaleAbs(usm_2, alpha = alpha, beta = beta)
# cv2.imshow('abjusted', adjusted)
#
# # 轉灰階
# grayscale = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayscale', grayscale)
#
# # 膨脹和腐蝕，去除黑白點
# kernel = np.ones((2, 2), np.uint8)
# swell = cv2.dilate(grayscale, kernel, iterations = 1) #膨脹
# corrosion = cv2.erode(swell, kernel, iterations = 1) #腐蝕
# swell_2 = cv2.dilate(corrosion, kernel, iterations = 1) #二次膨脹
# cv2.imshow('swell', swell)
# cv2.imshow('corrosion', corrosion)
# cv2.imshow('swell_2', swell_2)
#
# # 圖片二值化
# ret, binary = cv2.threshold(swell_2, 118, 255, cv2.THRESH_BINARY) #把圖片二值化，()內參數分別是：轉檔圖片名稱、
# cv2.imshow('binary', binary)
#
# # 改變圖片大小
# crop_x, crop_y = binary.shape
# # print(crop_x,'\n' , crop_y)
# magnification_x = 1.5
# magnification_y = 1.5
# enlarge = cv2.resize(binary, (int(crop_y * magnification_y), int(crop_x * magnification_x)))
# cv2.imshow('enlarge', enlarge)
#
# # 膨脹和腐蝕，去除黑白點
# # swell_3 = cv2.dilate(enlarge, kernel, iterations = 1) #膨脹
# # cv2.imshow('swell_3', swell_3)
#
# # 辨識圖片文字
# text = pytesseract.image_to_string(enlarge, lang='eng')  # 辨識圖片中文字部分
# text_data = pytesseract.image_to_data(enlarge, lang='eng')  # 辨識圖片中文字部分
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) #紀錄現在的時間
# if text == '':
#     print('沒有辨識出結果\n')
#     print(text_data)
#     cv2.imwrite("text_recognition_photo\\test\\fail\\camara_fail_" + now + ".png", enlarge)  # 儲存匯入模組的圖片
# else:
#     print(text)
#     print('\n' + text_data)
#     cv2.imwrite("text_recognition_photo\\test\\success\\camara_success" + now + ".png", enlarge)  # 儲存匯入模組的圖片
# cv2.waitKey(0)