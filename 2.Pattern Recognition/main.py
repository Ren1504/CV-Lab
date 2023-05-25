# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv.imread('messi.jpg', cv.IMREAD_GRAYSCALE)

# img2 = img.copy()
# template = cv.imread('ball.jpg', cv.IMREAD_GRAYSCALE)

# w, h = template.shape[::-1]

# # All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)

#     # Apply template Matching

#     res = cv.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc

#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)

#     cv.rectangle(img,top_left, bottom_right, 255, 2)

#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

#     plt.suptitle(meth)
#     plt.show()

import cv2

# Load the pattern and the image
pattern = cv2.imread('ten.png')  # Grayscale pattern image
image = cv2.imread('template.png')  # Grayscale input image

# Create a feature detector
orb = cv2.ORB_create()

# Find keypoints and descriptors in the pattern and the image
kp1, des1 = orb.detectAndCompute(pattern, None)
kp2, des2 = orb.detectAndCompute(image, None)

# Create a brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the descriptors
matches = bf.match(des1, des2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 10 matches
result = cv2.drawMatches(pattern, kp1, image, kp2, matches[:5], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the result
cv2.imshow('Pattern Recognition', result)
cv2.waitKey(0)
# cv2.destroyAllWindows()