import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0) #only one camera connected
if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20 #gives a little bit of leeway in the popup hand box, better for classifier
imgSize = 300

# only save when 's' is clicked
path = "data/D"
counter = 0

labels = ["A", "B", "C", "D"]

while True:
    success, img = cap.read() #capture frame by frame; success = True if frame read correctly
    hands, img = detector.findHands(img)
    if hands:
        #cropping the image based on the hand
        handFound = hands[0]
        x,y,w,h = handFound['bbox']

        #multiply by 255 to make it white
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 

        #an image is a matrix, give it starting x,y and ending x,y using
        #coordinates from the bounding box
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset] 

        #if h > w, match height to max size and adjust, and vice versa
        if h > w:
            w = math.ceil(imgSize * w / h)
            h = imgSize
            imgResize = cv2.resize(imgCrop, (w,h))
            wGap = math.ceil((imgSize - w) / 2) #gap needed to center image
            imgWhite[:, wGap:wGap+imgResize.shape[1]] = imgResize

            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)
        else:            
            h = math.ceil(imgSize * h / w)
            w = imgSize
            imgResize = cv2.resize(imgCrop, (w,h))
            hGap = math.ceil((imgSize - h) / 2) #gap needed to center image
            imgWhite[hGap:hGap+imgResize.shape[0], :] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) #1ms delay
