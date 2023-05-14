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
num = 0
while(cap_left.isOpened() and cap_right.isOpened()):    
    # Parameters from all steps are defined here to make it easier to adjust values.
    resolution     = 0.8    # (0, 1.0] the resolution of the new frame comparing to the old one
    numDisparities = 100     # has to be dividable by 16
    blockSize      = 7      # (0, 25] the block for measuring the similiarity
    windowSize     = 7      # Usually set equals to the block size
    filterCap      = 63     # [0, 100]
    lmbda          = 80000  # [80000, 100000]
    sigma          = 1.2
    brightness     = 0      # [-1.0, 1.0] Additional brightness for the final image
    contrast       = 0.7     # [0.0, 3.0] Additional contrast for the final image
    speckle_range = 500
    AVG_RATIO = 0.025
    WINDOW_COLOR = (0, 255, 255)
    THICKNESS = 5
    B = 0.1
    fx = 1400
    
    # Step 1 - Input the stream frame
    success_left, frame_left0 = cap_left.read()
    success_right, frame_right0 = cap_right.read()

    # Step 2 - Convert input to grayscale images and calibrate camera
    # Undistort and retify images 
    # frame_left0 = cv2.remap(frame_left0, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    # frame_right0 = cv2.remap(frame_right0, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.cvtColor(frame_left0,cv2.COLOR_BGR2GRAY)
    frame_right = cv2.cvtColor(frame_right0,cv2.COLOR_BGR2GRAY)
    frame_left = cv2.GaussianBlur(frame_left, (5,5), 0)
    frame_right = cv2.GaussianBlur(frame_right, (5,5), 0)
    height, width = frame_left.shape[:2]
    
    
    
    # Step 3 - Downsampling the images to the resolution level to speed up the matching at the cost of quality degradation.
    resL = cv2.resize(frame_left, None, fx = resolution, fy = resolution, interpolation = cv2.INTER_AREA)
    resR = cv2.resize(frame_right, None, fx = resolution, fy = resolution, interpolation = cv2.INTER_AREA)

    # Step 4 - Setup two stereo matchers to compute disparity maps both for left and right views.
    min_disparity=0
    max_disparity=64
    num_disp=max_disparity-min_disparity
    block_size=7
    uniqueness=7
    speckle_window_size=1000
    speckle_range=200
    left_matcher = cv2.StereoSGBM_create(
        numDisparities = num_disp,
        blockSize = block_size,
        uniquenessRatio = uniqueness,
        # mode = cv2.StereoSGBM_MODE_HH,
        speckleWindowSize = speckle_window_size,
        speckleRange = speckle_range,
        preFilterCap = 63
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
    imgLb = cv2.copyMakeBorder(frame_left, top = 0, bottom = 0, left = np.uint16(num_disp / resolution), right = 0, borderType= cv2.BORDER_CONSTANT, value = [155,155,155])
    filteredImg = wls_filter.filter(disparityL, imgLb, None, disparityR)
    # filteredImg = disparityL
    # Step 8 - Adjust image resolution, brightness, contrast, and perform disparity truncation hack
    filteredImg = filteredImg * resolution
    filteredImg = filteredImg + (brightness / 100.0)
    filteredImg = (filteredImg - 128) * contrast + 128
    filteredImg = np.clip(filteredImg, 0, 255)
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.resize(filteredImg, (width, height), interpolation = cv2.INTER_CUBIC) # Disparity truncation hack
    filteredImg = filteredImg[0:height, np.uint16(num_disp / resolution):width]
    filteredImg = cv2.resize(filteredImg, (width, height), interpolation = cv2.INTER_CUBIC)  # Disparity truncation hack

    saved_disp = np.array(filteredImg)
    # Step 9 - Calculate the distance
    min_disp = filteredImg.min()
    max_disp = filteredImg.max()

    if min_disp == max_disp:
        distance = 0
    else:
        # Calculate the average disparity of the center of the image
        center = np.mean(filteredImg[int(height*(0.5-AVG_RATIO)):int(height*(0.5+AVG_RATIO)), int(width*(0.5-AVG_RATIO)):int(width*(0.5+AVG_RATIO))])
        center_disp = (center - min_disp) / 64
        if center_disp <= 0:
            distance = 0
        else:
            distance = B * fx / center_disp

    # Print the output to the screen
    filteredImg = cv2.cvtColor(filteredImg, cv2.COLOR_GRAY2RGB)
    filteredImg = cv2.rectangle(filteredImg, (int(width*(0.5-AVG_RATIO)), int(height*(0.5-AVG_RATIO))), (int(width*(0.5+AVG_RATIO)), int(height*(0.5+AVG_RATIO))), WINDOW_COLOR, THICKNESS)
    filteredImg = cv2.line(filteredImg, (int(width/2), int(height*(0.5-0.05))), (int(width/2), int(height*(0.5+0.05))), WINDOW_COLOR,2)
    filteredImg = cv2.line(filteredImg, (int(width*(0.5-0.05)), int(height/2)), (int(width*(0.5+0.05)), int(height/2)), WINDOW_COLOR,2)
    PRINT_DISTANCE = 'Estimated distance: ' + str(int(distance))
    
    if (distance<=50):
        filteredImg = cv2.putText(filteredImg, PRINT_DISTANCE, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
        filteredImg = cv2.putText(filteredImg, "COLISION ALERT!!", (int(0.75*height), int(0.25*width)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        filteredImg = cv2.putText(filteredImg, PRINT_DISTANCE, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2, cv2.LINE_AA)

    
    
    # Hit 'q' to close the window or hit 's' to save

    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite('images/stereoLeft/o3d_input/imageL' + str(num) + '.png', frame_left)
        cv2.imwrite('images/stereoRight/o3d_input/imageR' + str(num) + '.png', frame_right)
        cv2.imwrite('images/disp_map/disp_map' + str(num) + '.png', saved_disp)
        print("Image and disparity map saved")
    elif k == ord('p'):
        break
    
    # Show the frames and result
    cv2.imshow('Left Cam', frame_left)
    cv2.imshow('Right Cam', frame_right)
    cv2.imshow('Disparity', filteredImg)
    
# Release and destroy all windows before determination
cap_left.release()
cap_right.release()

cv2.destroyAllWindows()