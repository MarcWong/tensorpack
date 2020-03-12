# Author: Tao Hu <taohu620@gmail.com>

import cv2
img = cv2.imread("1282.png")
edge = cv2.Canny(img, 100, 200)
cv2.imwrite("edge.png",edge)