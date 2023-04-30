import numpy as np
import cv2


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open both cameras
cap_left = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Direct show
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)

while(cap_left.isOpened() and cap_right.isOpened()):
    success_left, frame_left = cap_left.read()
    success_right, frame_right = cap_right.read()
    
    # Black white transformation
    frame_left = cv2.cvtColor(frame_left,cv2.COLOR_BGR2GRAY)
    frame_right = cv2.cvtColor(frame_right,cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur
    frame_left = cv2.GaussianBlur(frame_left, (3,3), 0)
    frame_right = cv2.GaussianBlur(frame_right, (3, 3), 0)
    # # # Undistort and retify images
    # frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    # frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    # Gray the frame
    min_disparity=0
    max_disparity=30
    num_disp=max_disparity-min_disparity
    block_size=7
    uniqueness=5
    speckle_window_size=1000
    speckle_range=100
    # p1=216
    # p2=864
    

    # Create Block matching object. 
    stereo = cv2.StereoSGBM_create(minDisparity= min_disparity,
        numDisparities = num_disp,
        blockSize = block_size,
        uniquenessRatio = uniqueness,
        mode = cv2.StereoSGBM_MODE_HH,
        speckleWindowSize = speckle_window_size,
        speckleRange = speckle_range,
        # disp12MaxDiff = max_disparity,
        # P1 = 8*1*block_size**2,#8*img_channels*block_size**2,
        # P2 = 32*1*block_size**2, #32*img_channels*block_size**2)
        )
    
    disparity = stereo.compute(frame_left, frame_right).astype('float32')/1024.0
    depth_map = 50.0/disparity
    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save
        cv2.imwrite('images/stereoLeft/image/imageL0' + str(num) + '.png', frame_left)
        cv2.imwrite('images/stereoRight/image/imageR0' + str(num) + '.png', frame_right)
        cv2.imwrite('images/disparity0'+str(num)+'.png', disparity)
        print('Image saved')
        num+=1
    # Show the frame
    cv2.imshow('Left Cam', frame_left)
    cv2.imshow('Right Cam', frame_right)
    cv2.imshow('Disparity', disparity)
    # Show the result
    # Hit 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break
    
# Release and destroy all windows before determination
cap_left.release()
cap_right.release()

cv2.destroyAllWindows()