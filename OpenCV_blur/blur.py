import cv2 as cv

# Sharp
img = cv.imread('data/IMG_0214.jpg', cv.IMREAD_GRAYSCALE)
cv.imshow('Sharp',img)
laplacian_var = cv.Laplacian(img, cv.CV_64F).var()
print(laplacian_var)

# Blur
img = cv.imread('data/IMG_0213.jpg', cv.IMREAD_GRAYSCALE)
cv.imshow('Blur',img)
laplacian_var = cv.Laplacian(img, cv.CV_64F).var()
print(laplacian_var)

if laplacian_var < 20:
    print("Image blurry")
else:
    print("Image okay")


cv.waitKey(0)