import cv2 as cv
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + \
    'haarcascade_frontalcatface_extended.xml')
import numpy as np

from os import listdir
from os.path import isfile, join


## 3/13/2021 7pm Applied on bad_bare and good_bare folders
def info_to_csv_backup(bad_or_good):
    # Rename files
    folder = 'frame_'+bad_or_good+'_rename/'
    mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_blur/'+folder

    # bare folders (without bounding box and rename)
    folder = bad_or_good+'_bare/'
    mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_blur/'+folder

    # Read all files in folder as a list
    files = sorted([f for f in listdir(mypath) if f!='.DS_Store'])


    # Headers (=column names for df)
    headers = ['filename','lp_cat','lp_all','lp_ratio','lp_cat_canny','lp_all_canny','lp_ratio_canny','blur','to_ctr','cat_x','cat_y','size_ratio']
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


        # Canny edges
        canny = cv.Canny(img,100,200)
        # Crop canny
        cropped_canny = canny[y:y+h,x:x+w]
        # cv.imshow('Cropped',cropped_canny)
        # cropped_gray = cv.cvtColor(cropped, cv.IMREAD_GRAYSCALE)
        kernel_size = 3;
        lp_cat_canny = cv.Laplacian(cropped_canny, cv.CV_64F,kernel_size).var()
        lp_all_canny = cv.Laplacian(canny, cv.CV_64F,kernel_size).var()
        lp_ratio_canny = lp_cat_canny/lp_all_canny



        # Cat face info
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
        cat_x = x/img.shape[1]
        cat_y = y/img.shape[0]



        rows.append([file,size_ratio,to_ctr])



        rows.append([file,lp_cat,lp_all,lp_ratio,lp_cat_canny,lp_all_canny,lp_ratio_canny,blur, to_ctr,cat_x,cat_y,size_ratio])
        cv.waitKey(1)


    # # Save the results as csv file
    # import csv
    # save_filename = 'blurriness_'+bad_or_good+'.csv'
    # with open(save_filename, 'w') as f:
    #     write = csv.writer(f)
    #     write.writerow(headers)
    #     write.writerows(rows)

    
    # Test with bare files
    # Save the results as csv file
    import csv
    save_filename = 'blurriness_'+bad_or_good+'_bare.csv'
    with open(save_filename, 'w') as f:
        write = csv.writer(f)
        write.writerow(headers)
        write.writerows(rows)
    


def info_to_csv(bad_or_good):
    # Rename files
    folder = 'frame_'+bad_or_good+'_rename/'
    mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_blur/'+folder

    # bare folders (without bounding box and rename)
    folder = bad_or_good+'_bare/'
    mypath = '/Users/katiehuang/Documents/best_cat/OpenCV_blur/'+folder


    # Read all files in folder as a list
    files = sorted([f for f in listdir(mypath) if f!='.DS_Store'])

    # YOLO detection
    # Load Yolo
    net = cv.dnn.readNet("../YOLO_detection/yolov3_custom_last.weights", "../YOLO_detection/yolov3_testing.cfg")

    # Name custom object
    # classes = ["Koala"]
    classes = ["cat_eye", "cat_ear", "cat_nose"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    # Headers (=column names for df)
    headers = ['filename','lp_cat','lp_all','lp_ratio','lp_cat_canny','lp_all_canny','lp_ratio_canny','blur',\
               'to_ctr','cat_x','cat_y','face_size','size_ratio',\
               'eyes','ears','nose']
    # Empty list to record each row (file)
    # Will be a list of lists
    rows = []

    # Empty list to record ROI/Whole value
    # lp_ratio = []
    # Empty list to record blurry or not (0: clear, 1: blurry)
    # blur = []

    files=['IMG_0647.jpg']
    # Read file
    for file in files[:]:
        print(file)
        # img = cv.imread(mypath+file, cv.IMREAD_GRAYSCALE)
        img = cv.imread(mypath+file)

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


        # Canny edges
        canny = cv.Canny(img,100,200)
        # Crop canny
        cropped_canny = canny[y:y+h,x:x+w]
        # cv.imshow('Cropped',cropped_canny)
        # cropped_gray = cv.cvtColor(cropped, cv.IMREAD_GRAYSCALE)
        kernel_size = 3;
        lp_cat_canny = cv.Laplacian(cropped_canny, cv.CV_64F,kernel_size).var()
        lp_all_canny = cv.Laplacian(canny, cv.CV_64F,kernel_size).var()
        lp_ratio_canny = lp_cat_canny/lp_all_canny



        # Cat face info
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
        cat_x = x/img.shape[1]
        cat_y = y/img.shape[0]



        
        # YOLO detection
        # Loading image
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    # Object detected
                    # print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(indexes)
        font = cv.FONT_HERSHEY_SIMPLEX

        eyes = []
        ears = []
        nose = []

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                # print(label,x,y,w,h)
                if label == "cat_eye":
                    eyes.append((x,y,w,h))
                elif label == "cat_ear":
                    ears.append([x,y,w,h])
                elif label == "cat_nose":
                    nose.append([x,y,w,h])

                color = colors[class_ids[i]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y + h +20), font, 1, color, 2)


        cv.imshow("Image", img)

        rows.append([file,lp_cat,lp_all,lp_ratio,lp_cat_canny,lp_all_canny,lp_ratio_canny,blur, to_ctr,cat_x,cat_y,cat_size,size_ratio,eyes,ears,nose])
        cv.waitKey(0)


    # # Save the results as csv file
    # import csv
    # save_filename = 'blurriness_'+bad_or_good+'.csv'
    # with open(save_filename, 'w') as f:
    #     write = csv.writer(f)
    #     write.writerow(headers)
    #     write.writerows(rows)

    
    # Test with bare files
    # Save the results as csv file
    import csv
    save_filename = 'TEST_face_features_'+bad_or_good+'_bare_portrait.csv'
    with open(save_filename, 'w') as f:
        write = csv.writer(f)
        write.writerow(headers)
        write.writerows(rows)




# info_to_csv('bad')
info_to_csv('good')