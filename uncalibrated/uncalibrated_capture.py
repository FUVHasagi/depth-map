import cv2

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():
    
    success, img = cap.read()
    
    k = cv2.waitKey(5)
    
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save
        if num%2==0:
            cv2.imwrite('uncalibrated\imgL\\imgL' + str(num//2) + '.png', img)
        else:
            cv2.imwrite('uncalibrated\imgR\\imgR' + str(num//2) + '.png', img)
        # cv2.imwrite('images/stereoLeft/non_calib/imageL' + str(num) + '.png', img1)
        # cv2.imwrite('images/stereoRight/non_calib/imageR' + str(num) + '.png', img2)
        print('Image saved')
        num+=1
    cv2.imshow('Img', img)

cap.release()

cv2.destroyAllWindows()