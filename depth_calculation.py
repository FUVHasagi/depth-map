import cv2
import numpy as np

min_disp = filtered_img.min()
max_disp = filtered_img.max()

if min_disp == max_disp:
    distance = 0
else:
    # Calculate the average disparity of the center of the image
    center = filtered_img[int(height/2), int(width/2)]
    center_disp = (center - min_disp) / 16
    if center_disp <= 0:
        distance = 0
    else:
        distance = baseline * fx / center_disp