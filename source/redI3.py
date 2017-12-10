from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from matplotlib.widgets import Button
from skimage import morphology
import math

def met(eyemat):
    r,g,b = cv2.split(eyemat)
    r = r.astype(np.float64)
    g = g.astype(np.float64)
    b = b.astype(np.float64)
    metric = (r**2)/(b**2+g**2+14)
    return metric

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))

def redI(image):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("face_model.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    maskl = np.ones_like(gray)*255
    maskr = np.ones_like(gray)*255

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    rects = detector(gray, 0)
    rcount = 0
    for rect in rects:
        rcount+=1
    if rcount!=0:    
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        maskl = np.zeros_like(gray)
        maskr = np.zeros_like(gray)

        cv2.drawContours(maskl, [leftEyeHull], 0, (255), -1)
        cv2.drawContours(maskr, [rightEyeHull], 0, (255), -1)

    masklr = cv2.bitwise_or(maskl,maskr)
    eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    eyeRects = eyesCascade.detectMultiScale(image , 1.1, 5)
    outImage = image.copy()

    for x,y,w,h in eyeRects:

        #Crop the eye region
        eyeImage = image [y:y+h , x:x+w]

        #split the images into 3 channels
        r, g ,b = cv2.split(eyeImage)
        # r = r.astype(np.int64)
        # g = g.astype(np.int64)
        # b = b.astype(np.int64)
        # Add blue and green channels
        bg = cv2.add(b,g)
        # bg = b**2+g**2+14
        #threshold the mask based on red color and combination ogf blue and gree color
        mask  = ( ((r)>(bg)) & (r>80) ).astype(np.uint8)*255
        #Some extra region may also get detected , we find the largest region
        masklr_roi = masklr[y:y+h , x:x+w]
        # print(masklr_roi.shape,mask.shape)
        mask = cv2.bitwise_and(mask,masklr_roi)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_DILATE,(5,5)) )
        mask = cv2.dilate(mask , (3,3) ,iterations=3)
        if np.count_nonzero(mask)==0:
            continue
        # eye_dat = cv2.bitwise_and(mask,eyeImage[:,:,2])
        metimg = met(eyeImage)

        meanI,stdI = weighted_avg_and_std(metimg, weights=mask)
        H,W = mask.shape
        xmesh,ymesh = np.meshgrid(np.arange(W),np.arange(H))
        wts = np.exp(-((metimg-meanI)**2)/(2.0*stdI*stdI))
        # print(meanI,stdI)
        wtst = wts>0.1
        meanx,stdx = weighted_avg_and_std(xmesh, weights=wtst)
        meany,stdy = weighted_avg_and_std(ymesh, weights=wtst)
        wtsg = np.exp(-((xmesh-meanx)**2)/(2.0*stdx*stdx)-((ymesh-meany)**2)/(2.0*stdy*stdy))
        
        # fwt = np.multiply(wts,wtsg)
        fwt = wtsg
        fwt = fwt/fwt.max()
        fwt[fwt<0.5] = 0
        mask = cv2.dilate(mask , cv2.getStructuringElement(cv2.MORPH_DILATE,(3,3)) ,iterations=1)
        # cv2.imwrite("mask_n.png",mask)
        fwt[mask==0] = 0
        mean  = bg /2
        

        # mean = cv2.bitwise_and(mean , mask )  # mask the mean image
        # mean  = cv2.cvtColor(mean ,cv2.COLOR_GRAY2BGR ) # convert mean to 3 channel
        # mask = cv2.cvtColor(mask ,cv2.COLOR_GRAY2BGR )  #convert mask to 3 channel
        # fwt = cv2.cvtColor(fwt ,cv2.COLOR_GRAY2BGR )
        # eye = cv2.bitwise_and(~mask,eyeImage)+mean           #Copy the mean color to masked region to color image
        eye = eyeImage.copy()
        eye[:,:,0] = fwt*mean + (1-fwt)*eyeImage[:,:,0]
        outImage [y:y+h , x:x+w] = eye
        plt.imsave("redI",outImage)
    return outImage


