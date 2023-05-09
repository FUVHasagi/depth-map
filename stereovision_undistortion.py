import numpy as np
import cv2

# Reference: https://github.com/pairote/stereo2depth/blob/master/stereo2depth.py

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open both cameras
cap_left = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Direct show
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)

while(cap_left.isOpened() and cap_right.isOpened()):    
    # Parameters from all steps are defined here to make it easier to adjust values.
    resolution     = 1.0    # (0, 1.0] the resolution of the new frame comparing to the old one
    numDisparities = 16     # has to be dividable by 16
    blockSize      = 7      # (0, 25] the block for measuring the similiarity
    windowSize     = 7      # Usually set equals to the block size
    filterCap      = 63     # [0, 100]
    lmbda          = 80000  # [80000, 100000]
    sigma          = 1.2
    brightness     = 0      # [-1.0, 1.0] Additional brightness for the final image
    contrast       = 1      # [0.0, 3.0] Additional contrast for the final image

    # Step 1 - Input the stream frame
    success_left, frame_left = cap_left.read()
    success_right, frame_right = cap_right.read()

    # Step 2 - Convert input to grayscale images.
    frame_left = cv2.cvtColor(frame_left,cv2.COLOR_BGR2GRAY)
    frame_right = cv2.cvtColor(frame_right,cv2.COLOR_BGR2GRAY)
    height, width = frame_left.shape[:2]

    # Step 3 - Downsampling the images to the resolution level to speed up the matching at the cost of quality degradation.
    resL = cv2.resize(frame_left, None, fx = resolution, fy = resolution, interpolation = cv2.INTER_AREA)
    resR = cv2.resize(frame_right, None, fx = resolution, fy = resolution, interpolation = cv2.INTER_AREA)

    # Step 4 - Setup two stereo matchers to compute disparity maps both for left and right views.
    left_matcher = cv2.StereoSGBM_create(
        minDisparity = 0,
        numDisparities = numDisparities,
        blockSize = blockSize,
        P1 = 8 * 3 * windowSize ** 2,
        P2 = 32 * 3 * windowSize ** 2,
        disp12MaxDiff = 1,
        uniquenessRatio = 15,
        speckleWindowSize = 0,
        speckleRange = 2,
        preFilterCap = filterCap,
        mode = cv2.STEREO_SGBM_MODE_HH
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Step 5 - Setup a disparity filter to deal with stereo-matching errors.
    #          It will detect inaccurate disparity values and invalidate them, therefore making the disparity map semi-sparse
    #          Beside the WLS Filter has an excellent performance on edge preserving smoothing technique
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Step 6 - Perform stereo matching to compute disparity maps for both left and right views.
    disparityL = left_matcher.compute(resL, resR)
    disparityR = right_matcher.compute(resR, resL)

    # Step 7 - Perform post-filtering
    imgLb = cv2.copyMakeBorder(frame_left, top = 0, bottom = 0, left = np.uint16(numDisparities / resolution), right = 0, borderType= cv2.BORDER_CONSTANT, value = [155,155,155])
    filteredImg = wls_filter.filter(disparityL, imgLb, None, disparityR)
    
    # Step 8 - Adjust image resolution, brightness, contrast, and perform disparity truncation hack
    filteredImg = filteredImg * resolution
    filteredImg = filteredImg + (brightness / 100.0)
    filteredImg = (filteredImg - 128) * contrast + 128
    filteredImg = np.clip(filteredImg, 0, 255)
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.resize(filteredImg, (width, height), interpolation = cv2.INTER_CUBIC) # Disparity truncation hack
    filteredImg = filteredImg[0:height, np.uint16(numDisparities / resolution):width]
    filteredImg = cv2.resize(filteredImg, (width, height), interpolation = cv2.INTER_CUBIC)  # Disparity truncation hack
    
    # Show the frame
    cv2.imshow('Left Cam', frame_left)
    cv2.imshow('Right Cam', frame_right)
    cv2.imshow('Disparity', filteredImg)
    # Show the result
    # Hit 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break
    
# Release and destroy all windows before determination
cap_left.release()
cap_right.release()

cv2.destroyAllWindows()