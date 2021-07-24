import numpy as np
import cv2


fcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
#Using xml files from OpenCV Git

video=cv2.VideoCapture(0)

while True:
    ret, frame= video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face=fcascade.detectMultiScale(gray,1.1,4)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #Making rectangles after detection

    
    font=cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame,'Divyansh Agarwal',(10,400), font,2,(0,255,0))
    cv2.line(frame, (50,20) , (500,20) , (255,255,255) , 2 )
    cv2.line(frame, (75,30) , (475,30) , (255,255,255) , 1 )
    cv2.rectangle(frame, (85,50) , (450,300) , (0,255,0) , 4 )
    #Making drawings

    cv2.imshow('Video', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()