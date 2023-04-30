import cv2

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

num = 0

while cap1.isOpened() and cap2.isOpened():
    
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()
    
    k = cv2.waitKey(5)
    
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save
        cv2.imwrite('images/stereoLeft/image/imageL' + str(num) + '.png', img1)
        cv2.imwrite('images/stereoRight/image/imageR' + str(num) + '.png', img2)
        print('Image saved')
        num+=1
    cv2.imshow('Img 1', img1)
    cv2.imshow('Img 2', img2)

cap1.release()
cap2.release()

cv2.destroyAllWindows()