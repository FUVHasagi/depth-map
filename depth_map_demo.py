import cv2
import numpy as np

imgL = cv2.imread('/images/stereoLeft/image/')
imgR = cv2.imread('/images/stereoRight/image/')

stereo = cv2.StereoBM_create(numDisparities=16, blockSize = 15)
disparity = stereo.compute(imgL, imgR)
cv2.imshow('Depth', disparity)
cv2.waitKey(10000)
cv.destroyAllWindows()