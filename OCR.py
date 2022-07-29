import cv2
from PIL import Image
import pytesseract
import plot
import np
import os, sys
import csv
import time
from difflib import SequenceMatcher

# ----------存放正確文字資料
text1 = '14526861-1\n2013041320\n06-04-2013\nMADE IN TAIWAN' #test20_1
text2 = 'ASSY 14526561-1\n2013060606\n06-11-2013\nMADE IN TAIWAN' #test20_2
text3 = '14526862-1\n2013120219\n01-21-2014\nMADE IN TAIWAN' #test21_1

# ----------存放前處理設定資料
sigma_1 = 30 #第一次銳化參數
sigma_2 = 80 #第二次銳化參數
brightness = 1.7 #亮度
contrast = 0.4 #對比度
kernel = [[1, 1], [2, 2], [2, 2]] #膨脹、腐蝕共用參數
dilate_1 = [4, 1, 1] #第一次膨脹次數
erode = [1, 1, 2] #腐蝕次數
dilate_2 = [4, 1, 2] #地二次膨脹次數
binarization = [[], [], [], []] #二值化參數

# ----------輸入測試出的二值化參數
for i in range(4, 23):
    binarization[0].append(i)
for i in range(37, 41):
    binarization[1].append(i)
for i in range(108, 122):
    binarization[2].append(i)
for i in range(135, 138):
    binarization[3].append(i)
print('binarization：', binarization)

def OCR(img):

    # ----------圖像前處理
    text = '14526362-\n201312021\n\nOi-gl-20j4\nMADE IN Ta twat' #辨識文字內容
    m = SequenceMatcher(None, text2, text)
    correct_rate = m.ratio()
    print(correct_rate)