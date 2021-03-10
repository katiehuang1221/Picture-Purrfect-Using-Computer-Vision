import cv2 as cv
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + \
    'haarcascade_frontalcatface_extended.xml')

from os import listdir
from os.path import isfile, join

# Bad frames
# Read all files in folder as a list
mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_blur/frame_bad_rename/'
files = sorted([f for f in listdir(mypath) if (isfile(join(mypath, f))) & (f != ".DS_Store")])
# print('Number of files:', len(files))
# print(files)

# Headers (=column names for df)
headers = ['filename','size_ratio','to_ctr']
# Empty list to record each row (file)
# Will be a list of lists
rows = []

# # Empty list to record ROI/Whole value
# # lp_ratio = []
# # Empty list to record blurry or not (0: clear, 1: blurry)
# # blur = []
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

    img_size = img.shape[0]*img.shape[1]
    # print('Size of image:',img_size)
    cat_size = w*h
    # print('Size of cat face:',cat_size)
    size_ratio = cat_size/img_size
    # print('Size ratio:', size_ratio)

    img_ctr = (img.shape[1]/2, img.shape[0]/2)
    cat_ctr = (x,y)
    # print('Center of image:', img_ctr)
    # print('Center of cat face:', cat_ctr)

    import math
    to_ctr = math.dist(img_ctr,cat_ctr)
    # print('Distance to center:', to_ctr)


    rows.append([file,size_ratio,to_ctr])

    cv.waitKey(1)


# Save the results as csv file
import csv
with open('face_info_bad.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(headers)
    write.writerows(rows)





# Good frames
# Read all files in folder as a list
mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_blur/frame_good_rename/'
files = sorted([f for f in listdir(mypath) if (isfile(join(mypath, f))) & (f != ".DS_Store")])
# print('Number of files:', len(files))
# print(files)

# Headers (=column names for df)
headers = ['filename','size_ratio','to_ctr']
# Empty list to record each row (file)
# Will be a list of lists
rows = []

# # Empty list to record ROI/Whole value
# # lp_ratio = []
# # Empty list to record blurry or not (0: clear, 1: blurry)
# # blur = []
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

    img_size = img.shape[0]*img.shape[1]
    # print('Size of image:',img_size)
    cat_size = w*h
    # print('Size of cat face:',cat_size)
    size_ratio = cat_size/img_size
    # print('Size ratio:', size_ratio)

    img_ctr = (img.shape[1]/2, img.shape[0]/2)
    cat_ctr = (x,y)
    # print('Center of image:', img_ctr)
    # print('Center of cat face:', cat_ctr)

    import math
    to_ctr = math.dist(img_ctr,cat_ctr)
    # print('Distance to center:', to_ctr)


    rows.append([file,size_ratio,to_ctr])

    cv.waitKey(1)


# Save the results as csv file
import csv
with open('face_info_good.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(headers)
    write.writerows(rows)