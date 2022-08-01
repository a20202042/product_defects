import cv2
import numpy as np
import math, imutils, json, os
from math import copysign, log10
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
from PIL import Image
import pytesseract, re

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4

ret, cv_img = cap.read()

while (True):
    # 擷取影像
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    status = cv2.imwrite('text_115.png', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
