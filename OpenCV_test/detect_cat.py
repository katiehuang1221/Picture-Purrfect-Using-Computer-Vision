import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('data_test/cat2.jpg')


# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# haar_cascade cat face classifier
cascade_classifier = cv.CascadeClassifier('haar_catface.xml')
faces_rect = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
print(f'Number of cat faces found = {len(faces_rect)}')


for (x,y,w,h) in faces_rect:
    cv.rectangle(rgb, (x,y), (x+w,y+h), (0,255,0), thickness=3)


# Default color in matplotlib is RGB
plt.imshow(rgb)
plt.show()

# save the image
plt.imsave('test.png', rgb)


cv.imshow('Detected Cat Faces', img)

cv.waitKey(0)