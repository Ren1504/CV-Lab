import cv2

img=cv2.imread("dragons.jpg")
cv2.imshow("Original",img)
cv2.waitKey(0)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_blur=cv2.GaussianBlur(gray,(3,3),0)

sobelx=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=5)
sobely=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=0,dy=1,ksize=5)
sobelxy=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=1,dy=1,ksize=5)

cv2.imshow("sobelx",sobelx)
cv2.waitKey(0)

cv2.imshow("sobely",sobely)
cv2.waitKey(0)

cv2.imshow("sobelxy",sobelxy)
cv2.waitKey(0)