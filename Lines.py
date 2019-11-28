#!/usr/bin/python3
#https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
import cv2
import numpy as np

def LineSegmentation(img, saveResults=True):

    ## (1) Gray Scale Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## (2) threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    
    
    ## (3) minAreaRect on the nozeros
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)
    box = cv2.boxPoints(ret) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    # cv2.drawContours(threshed,[box],0,(255,255,255),2)

    # print(ret)
    (cx,cy), (h,w), ang = ret

    
    # print(h,w,ang)
    if (ang <-45) :
        # w,h = h,w
        ang +=90
 


    ## (4) Find rotated matrix, do rotation
    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
    pts = cv2.findNonZero(rotated)
    ret = cv2.minAreaRect(pts)
    # print(ret)
    # cv2.imshow('original',threshed)
    # cv2.imshow('rotated',rotated)
    # cv2.waitKey(0)    
    
    # rotated=255-rotated
    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)
    # print(hist)
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if (y-4>0 and hist[y]==0 and hist[y-1]==0 and hist[y-2]==0 and hist[y-3]==0 and hist[y-4]==0 and hist[y+1]>0)]
    lowers = [y for y in range(H-1) if (y+4<H-1 and hist[y]>0 and hist[y+1]==0 and hist[y+2]==0 and hist[y+3]==0 and hist[y+4]==0)]

    
    # rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    # rotated=255-rotated
 
    lines = []

    for y in range(len(uppers)-1):
        lines += [rotated[uppers[y]:lowers[y]+1,:]]
        # if saveResults:
        cv2.imwrite("PreprocessingOutput/LineSegmentation/line"+str(y)+".png",rotated[uppers[y]:lowers[y]+1,:])
    
    # if saveResults:
    #     for y in uppers:
    #         cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

    #     for y in lowers:
    #         cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)

    #     cv2.imwrite("result.png", rotated)

    return lines

# LineSegmentation(cv2.imread("Dataset/scanned/capr377.png"),True)