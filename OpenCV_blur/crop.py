import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_catface.xml')

# Sharp
img = cv.imread('data/IMG_0214.PNG')
# img = cv.imread('data/IMG_0214.jpg', cv.IMREAD_GRAYSCALE)

# Blurry
# img = cv.imread('data/IMG_0213.PNG')



gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of cat faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=3)


cv.imshow('Detected Cat Faces', img)


# Crop
(x,y,w,h) = faces_rect[0]
cropped = img[y:y+h,x:x+w]
cv.imshow('Cropped',cropped)
cropped_gray = cv.cvtColor(cropped, cv.IMREAD_GRAYSCALE)
laplacian_var_roi = cv.Laplacian(cropped_gray, cv.CV_64F).var()
print("ROI:",laplacian_var_roi)
laplacian_var_whole = cv.Laplacian(gray, cv.CV_64F).var()
print("Whole:",laplacian_var_whole)


if laplacian_var_roi < laplacian_var_whole:
    print("Image blurry")
else:
    print("Image okay")



cv.waitKey(0)