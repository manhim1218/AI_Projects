import cv2
 
# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(4)
i = 0

count = 0
capture_frame = 10 

 
while(cap.isOpened()):
    ret, frame = cap.read()
     
    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break
     
    if count % capture_frame == 0:
    # Save Frame by Frame into disk using imwrite method
        cv2.imwrite('/home/irlcrossing/Desktop/Object detection/yolo_label/Apple/apple'+str(i)+'.jpg', frame)
        i += 1
 
cap.release()
cv2.destroyAllWindows()
