import cv2 as cv
import numpy as np
import os

# To read the images from the file path
path = 'ImagesQuery'
orb = cv.ORB_create(nfeatures=1000)
images = []  # List of the image
className = []  # Name of the images
myList = os.listdir(path)
print(myList)
print('Total classes detected', len(myList))

# To get the images from the ImageQuery Folder
for cl in myList:
    imgCur = cv.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])

print(className)

# This Loops through the stored images to find the Descriptor
def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

# This Loops takes the image displayed through the camera and returns the closest id from the train set
def findId(img, desList, thres= 15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try :
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
        print(matchList)
    except:
        pass

    if len(matchList)!= 0 :
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal



desList = findDes(images)
print(len(desList))

cap = cv.VideoCapture(0)

while True:

    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    id = findId(img2, desList,thres=15)
    if id != -1:
        cv.putText(imgOriginal, className[id], (50,50), cv.FONT_HERSHEY_SIMPLEX,1, (0, 0 ,255),1)


    cv.imshow('img2', imgOriginal)
    cv.waitKey(1)