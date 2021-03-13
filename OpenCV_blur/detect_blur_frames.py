import cv2 as cv
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + \
    'haarcascade_frontalcatface_extended.xml')

from os import listdir
from os.path import isfile, join

# Bad frames
# Read all files in folder as a list
mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_blur/frame_bad_rename/'
# files = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
files = sorted([f for f in listdir(mypath) if f!='.DS_Store'])
# print('Number of files:', len(files))
# print(files)

# Headers (=column names for df)
headers = ['filename','lp_cat','lp_all','lp_ratio','blur']
# Empty list to record each row (file)
# Will be a list of lists
rows = []

# Empty list to record ROI/Whole value
# lp_ratio = []
# Empty list to record blurry or not (0: clear, 1: blurry)
# blur = []
# Read file
for file in files[0:]:
    print(file)
    img = cv.imread(mypath+file, cv.IMREAD_GRAYSCALE)

    # Detect cat face
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    # Draw rectangle around detected cat face
    # print(len(faces_rect))
    if len(faces_rect) > 0:
        for (x,y,w,h) in faces_rect:
            cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=3)

        (x,y,w,h) = faces_rect[0]

    else:
        (x,y,w,h) = (0,0,1,1)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=3)


    cv.imshow('Detected Cat Faces', img)

    # Crop and leave cat face only
    # (x,y,w,h) = faces_rect[0]
    cropped = img[y:y+h,x:x+w]
    cv.imshow('Cropped',cropped)
    kernel_size=3;
    laplacian_var_roi = cv.Laplacian(cropped, cv.CV_64F,kernel_size).var()
    # print("ROI:",laplacian_var_roi)
    # cropped_gray = cv.cvtColor(cropped, cv.IMREAD_GRAYSCALE)
    # print(w,h)
    # print(img.shape)
    # print(img.shape[1]/w)
    scale=img.shape[1]//w

    resized = cv.resize(cropped, (w*scale,h*scale), interpolation=cv.INTER_CUBIC)
    cv.imshow('Resized',resized)
    lp_cat = cv.Laplacian(resized, cv.CV_64F,kernel_size).var()
    # print("ROI resized:",laplacian_var_roi_resized)

    lp_all = cv.Laplacian(img, cv.CV_64F,kernel_size).var()
    # print("Whole:",laplacian_var_whole)
    lp_ratio = lp_cat/lp_all
    # lp_ratio.append(laplacian_var_roi_resized/laplacian_var_whole)
    # print("ROI/Whole:",laplacian_var_roi/laplacian_var_whole)
    # print("ROIr/Whole:",laplacian_var_roi_resized/laplacian_var_whole)


    # Check if blurry
    if lp_cat/lp_all < 0.03:
        blur = 1
        # blur.append(1)
        # print("Image blurry")
    else:
        blur = 0
        # blur.append(0)
        # print("Image okay")

    rows.append([file,lp_cat,lp_all,lp_ratio,blur])

    cv.waitKey(1)


# Save the results as csv file
import csv
with open('blurriness_bad.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(headers)
    write.writerows(rows)



# Good frames
# Read all files in folder as a list
mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_blur/frame_good_rename/'
files = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
# print('Number of files:', len(files))
# print(files)

# Headers (=column names for df)
headers = ['filename','lp_cat','lp_all','lp_ratio','blur']
# Empty list to record each row (file)
# Will be a list of lists
rows = []

# Empty list to record ROI/Whole value
# lp_ratio = []
# Empty list to record blurry or not (0: clear, 1: blurry)
# blur = []
# Read file
for file in files[:]:
    print(file)
    img = cv.imread(mypath+file, cv.IMREAD_GRAYSCALE)

    # Detect cat face
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    # Draw rectangle around detected cat face
    for (x,y,w,h) in faces_rect:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=3)

    cv.imshow('Detected Cat Faces', img)

    # Crop and leave cat face only
    # (x,y,w,h) = faces_rect[0]
    cropped = img[y:y+h,x:x+w]
    cv.imshow('Cropped',cropped)
    kernel_size=3;
    laplacian_var_roi = cv.Laplacian(cropped, cv.CV_64F,kernel_size).var()
    # print("ROI:",laplacian_var_roi)
    # cropped_gray = cv.cvtColor(cropped, cv.IMREAD_GRAYSCALE)
    # print(w,h)
    # print(img.shape)
    # print(img.shape[1]/w)
    scale=img.shape[1]//w

    resized = cv.resize(cropped, (w*scale,h*scale), interpolation=cv.INTER_CUBIC)
    cv.imshow('Resized',resized)
    lp_cat = cv.Laplacian(resized, cv.CV_64F,kernel_size).var()
    # print("ROI resized:",laplacian_var_roi_resized)

    lp_all = cv.Laplacian(img, cv.CV_64F,kernel_size).var()
    # print("Whole:",laplacian_var_whole)
    lp_ratio = lp_cat/lp_all
    # lp_ratio.append(laplacian_var_roi_resized/laplacian_var_whole)
    # print("ROI/Whole:",laplacian_var_roi/laplacian_var_whole)
    # print("ROIr/Whole:",laplacian_var_roi_resized/laplacian_var_whole)


    # Check if blurry
    if lp_cat/lp_all < 0.03:
        blur = 1
        # blur.append(1)
        # print("Image blurry")
    else:
        blur = 0
        # blur.append(0)
        # print("Image okay")

    rows.append([file,lp_cat,lp_all,lp_ratio,blur])

    cv.waitKey(1)


# Save the results as csv file
import csv
with open('blurriness_good.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(headers)
    write.writerows(rows)