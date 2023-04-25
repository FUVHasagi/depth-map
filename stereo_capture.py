import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

num = 0

while cap.isOpened():
    
    success1, img = cap.read()
    success2, img2 = cap.read()
    
    k = cv2.waitKey(5)
    
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save
        cv2.imwrite('images/stereoLeft/image/imageL' + str(num))
        cv2.imwrite('images/stereoRight/image/image2' + str(num))
        print('Image saved')
        num+=1
    cv2.imshow('Img 1', img)
    cv2.imshow('Img 2', img2)

cap.release()
cap2.release()

cv2.destroyAllWindow()