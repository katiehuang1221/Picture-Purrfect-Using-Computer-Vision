import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_save(file):
    cascade_classifier = cv.CascadeClassifier('haar_cat_face.xml')
    
    folder='data_test/'
    filename=file
    file_type='.MOV'
    
    cap = cv.VideoCapture(folder+filename+file_type)
    i = 0
    while True:
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, 0)
        detections = cascade_classifier.detectMultiScale(frame, 1.3, 5)

        if (len(detections)>0):
            (x,y,w,h) = detections[0]
            frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv.imwrite('saved_frame/'+filename+'_'+str(i)+'.jpg',frame)
            
        cv.imshow('frame',frame)
        i+=1
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


files = ['IMG_0185','IMG_0186','IMG_0187']
detect_save(files[2])
