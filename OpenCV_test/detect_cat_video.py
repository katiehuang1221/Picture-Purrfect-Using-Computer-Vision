import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_save(file):
    cascade_classifier = cv.CascadeClassifier('haar_cat_face.xml')
    
    folder='data_video/'
    filename=file
    file_type='.MOV'
    
    cap = cv.VideoCapture(folder+filename)
    i = 0
    while True:
        ret, frame = cap.read()

        if ret: # Added on 3/9/2021: fixed ending error
            frame = cv.cvtColor(frame, 0)
            detections = cascade_classifier.detectMultiScale(frame, 1.3, 5)

            if (len(detections)>0):
                (x,y,w,h) = detections[0]
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                if i%10==0: # Added on 3/8/2021
                    cv.imwrite('saved_frames/'+filename+'_'+str(i)+'.jpg',frame)
                
            cv.imshow('frame',frame)
            i+=1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
        
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv.destroyAllWindows()



# Read video filenames
from os import listdir
from os.path import isfile, join

mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_test/data_video'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print('Number of files:', len(files))

# files = ['IMG_0185','IMG_0186','IMG_0187']
# detect_save(files[2])
for file in files:
    detect_save(file)
