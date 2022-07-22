import sys
import cv2 as cv
import numpy as np

#vvvvvv
def main(argv):
    default_file = '0.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray,
                              cv.HOUGH_GRADIENT,
                              minDist=6,
                              dp=1.1,
                              param1=150,
                              param2=15,
                              minRadius=8,
                              maxRadius=10)
    circles_data = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        number = 1
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
            circles_data.append({'number': number, 'center': center, 'r': i})
            number = number + 1
    print(circles_data)
    for item in circles_data:
        text = 'circle_%s' % item['number']
        cv.putText(src, text, item['center'], cv.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 0), 2, cv.LINE_AA)
    cv.imshow("detected circles", src)

    cv.waitKey(0)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
