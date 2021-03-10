import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_catface.xml')

# Sharp
# img = cv.imread('data/IMG_0214.PNG')
img = cv.imread('data/IMG_0214.PNG', cv.IMREAD_GRAYSCALE)

# Blurry
# img = cv.imread('data/IMG_0213.PNG')
# img = cv.imread('data/IMG_0213.PNG', cv.IMREAD_GRAYSCALE)


# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1)

print(f'Number of cat faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=3)


cv.imshow('Detected Cat Faces', img)


# Crop
(x,y,w,h) = faces_rect[0]
cropped = img[y:y+h,x:x+w]
cv.imshow('Cropped',cropped)
# cropped_gray = cv.cvtColor(cropped, cv.IMREAD_GRAYSCALE)
kernel_size = 101;
laplacian_var_roi = cv.Laplacian(cropped, cv.CV_64F,kernel_size).var()
print("ROI:",laplacian_var_roi)
laplacian_var_whole = cv.Laplacian(img, cv.CV_64F,kernel_size).var()
print("Whole:",laplacian_var_whole)
print("ROI/Whole",laplacian_var_roi/laplacian_var_whole)


if laplacian_var_roi/laplacian_var_whole < 0.9:
    print("Image blurry")
else:
    print("Image okay")


cv.waitKey(0)

