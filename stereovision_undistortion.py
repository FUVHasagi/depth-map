import numpy as np
import cv2


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open(stereoMap.xml, cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open both cameras
cap_left = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Direct show
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)

while(cap_left.isOpened() and cap_right.isOpened()):
    success_left, frame_left = cap_left.read()
    success_right, frame_right = cap_right.read()
    
    # Undistort and retify images
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    
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