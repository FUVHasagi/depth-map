import numpy as np
import cv2 as cv
import glob
#  FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS

chessboardSize = (9,6)
frameSize = (640, 480)

# terminate criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((chessboardSize[0]*chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

objp = objp*20
print(objp)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpointsL = []
imgpointsR = []

imagesLeft = glob.glob('image/stereoLeft/*.png')
imagesRight = glob.glob('images/stereoRight/*.png')

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    
    # Find the corners of the chessboard
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    resR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    
    # If found, add objects points, image points (after refining them)
    if retL and retR == True:
        objpoints.append(objp)
        
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1, -1), criteria)
        imgpointsL.append(cornersL)
        
        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        
        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('Image Left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('Image Right', imgR)
        cv.waitKey(1000)
        
cv.destroyAllWindows()


# 