from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import numpy as np

img = cv2.imread("homelander.png")

cv2.imshow("xqc",img)
cv2.waitKey(0)

grey = rgb2gray(img)
cv2.imshow("grey",grey)
cv2.waitKey(0)

gray_r = grey.reshape(grey.shape[0]*grey.shape[1])

for i in range(gray_r.shape[0]):
  gray_r[i] = 1 if gray_r[i] > gray_r.mean() else 0

final = gray_r.reshape(grey.shape[0],grey.shape[1])

cv2.imshow("output",final)
cv2.waitKey(0)
