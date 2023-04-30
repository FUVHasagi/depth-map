import cv2
import numpy as np
import matplotlib.pyplot as plt
imgL = cv2.imread('./images/stereoLeft/image/imageL2.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('./images/stereoRight/image/imageR2.png', cv2.IMREAD_GRAYSCALE)
print(imgL.shape)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize = 15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity,'gray')
plt.show()