import cv2
import numpy as np
import math, imutils, json, os
from math import copysign, log10
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob

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

ret, cv_img = cap.read()


def contour_calculation(size, frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    gray_res = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel=kernel)
    return gray_res


def box_normal(bgr):
    gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    contoure, he = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box = [cv2.boundingRect(cnt) for cnt in contoure]

    for bbox in bounding_box:
        [x, y, w, h] = bbox
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return bgr


def box_compact(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    img_2, img_3 = img.copy(), img.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
    # cv2.imshow('img', img)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_2, [box], 0, (0, 0, 255), 2)
    return img, img_2, box


def angle_calculate(box):
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
    return angle_


def rotate_img(img, angle):
    rotated_img = imutils.rotate_bound(img, angle)
    # rotated_img
    return rotated_img


def crop(left_point_x, right_point_x, top_point_y, bottom_point_y, crop_img):
    range = 5
    x = min([left_point_x, right_point_x]) - range
    w = abs(right_point_x - left_point_x) + range * 2
    y = min([top_point_y, bottom_point_y]) - range
    h = abs(top_point_y - bottom_point_y) + range * 2
    crop_img = crop_img[y:y + h, x:x + w]
    return crop_img


def circle_detect(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               minDist=6, dp=1.1,
                               param1=100, param2=15,
                               minRadius=8, maxRadius=50)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles_data = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        number = 1
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(crop_img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(crop_img, center, radius, (255, 0, 255), 3)
            circles_data.append({'number': number, 'center': center, 'r': i})
            number = number + 1
    for item in circles_data:
        text = 'circle_%s' % item['number']
        cv2.putText(crop_img, text, item['center'], cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2, cv2.LINE_AA)
    # print(len(circles_data))
    return crop_img, circles_data
    # cv2.imshow("detected circles", crop_img)


def line_detect(src):
    dst = cv2.Canny(src, 0, 255, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    # cv2.imshow("Source", src)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    return cdstP


def hu_moment(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(im)
    huMoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
        huMoments[i] = round(huMoments[i][0], 3)
    hu = []
    for item in huMoments:
        hu.append(float(item[0]))
    return hu


def load_hu_data():
    f = open('data.json')
    data = json.load(f)
    # print(data)
    return data


def compare_hu_moment(file, hu, hu_data):
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


def match_crop_img(img, x, y, w, h):
    crop_img = img[y:y + h, x:x + w]
    return crop_img


def similarity(item):
    model = SentenceTransformer('clip-ViT-B-32')
    image_names = list(glob.glob('.\\match_data\\*.png'))  # 圓圖資料夾
    image_names.append(item)
    # print(image_names)
    # print("Images:", len(image_names))
    encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128,
                                 convert_to_tensor=True, show_progress_bar=True)

    processed_images = util.paraphrase_mining_embeddings(encoded_image)
    NUM_SIMILAR_IMAGES = 10

    # duplicates = [image for image in processed_images if image[0] >= 0.999]
    #
    # for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
    #     print("\nScore: {:.3f}%".format(score * 100))
    #     print(image_names[image_id1])
    #     print(image_names[image_id2])

    threshold = 0.99
    near_duplicates = [image for image in processed_images if image[0] < threshold]

    source = {}
    for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
        print("\nScore: {:.3f}%".format(score * 100))
        print(image_names[image_id1])
        print(image_names[image_id2])
        if image_names[image_id2] == item:
            source.update({image_names[image_id1]: score * 100})
    print(source)
    return source


hu_data = load_hu_data()

dir = 'match_data'
files = os.listdir(dir)
file_data = []
for file in files:
    template = cv2.imread(dir + '\\' + file)
    file_data.append({'template': template, 'file': file})

while (True):

    # 擷取影像
    ret, frame = cap.read()
    rotated_img, crop_img = frame.copy(), frame.copy()
    cv2.imshow('cam', frame)

    gray_res = contour_calculation(2, frame)
    # cv2.imshow('gray_res', gray_res)

    box_img, box_compact_img, box = box_compact(frame)
    crop_img = crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]), crop_img)

    # # -----抓出最小方形角度並旋轉
    angle = angle_calculate(box)
    rotated_img = rotate_img(rotated_img, angle)
    # # cv2.imshow('rotated_img', rotated_img)
    #
    # # -----裁切出目標區域
    # not_use, not_use_2, box = box_compact(rotated_img)
    # crop_img = rotate_img(crop_img, angle)
    # crop_img = crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]), crop_img)

    try:
        # cv2.imshow('crop_img', crop_img)
        cv2.imshow('crop_img', crop_img)
        check = True
        # status = cv2.imwrite('match_data/or_3.png', crop_img)
    except:
        crop_img = np.zeros((500, 500, 3), dtype="uint8")
        check = False
        # print('crop_img not crop')  # 測試用

    # ------目標區域分類
    if check is True:
        for item in file_data:
            template = item['template']
            file = item['file']
            h, w, l = template.shape
            if crop_img.shape[0] > h and crop_img.shape[1] > w:
                res = cv2.matchTemplate(crop_img, template, cv2.TM_CCORR_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                max_val_ncc = '{:.3f}'.format(max_val)
                if float(max_val_ncc) > 0.96:
                    match = True
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    hu = hu_moment(crop_img)
                    if str(file) in hu_data.keys():
                        HU = compare_hu_moment(str(file), hu, hu_data)
                        # print(HU)
                        if HU is True:
                            hu_check = True
                            cv2.putText(frame, item['file'] + max_val_ncc + 'hu_go', [100, 100], cv2.FONT_HERSHEY_SIMPLEX,
                                        1,(255, 255, 0), 2, cv2.LINE_AA)
                            break
                        else:
                            cv2.putText(frame, item['file'] + max_val_ncc, [100, 100], cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 0), 2, cv2.LINE_AA)
                            break
                else:
                    cv2.putText(frame, ' ', [100, 100], cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'not_find', [100, 100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)

        cv2.imshow('cam', frame)

    # # ------目標區域圓孔偵測
    # circle_detect_img = crop_img.copy()
    # circle_detect_img, circles_data = circle_detect(circle_detect_img)
    # cv2.imshow('circle_detect_img', circle_detect_img)
    #
    # # ------目標區域線段偵測
    # line_target_img = crop_img.copy()
    # line_target_img = line_detect(line_target_img)
    # cv2.imshow('line_target_img', line_target_img)
    # cv2.imshow('crop_img', crop_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
