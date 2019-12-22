#!/usr/bin/python3
#https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def LineSegmentation(img, imgName = "", saveResults=True):

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
    # ################## Method 1 ##################
    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
    pts = cv2.findNonZero(rotated)
    ret = cv2.minAreaRect(pts)
    
    # ################## Method 2 ##################
    # skewImage = threshed
    # ht, wd = skewImage.shape
    # bin_img = 1 - (threshed.reshape((ht, wd)) / 255.0)

    # delta = 1
    # limit = 5
    # angles = np.arange(-limit, limit+delta, delta)
    # scores = []
    # for angle in angles:
    #     hist, score = find_score(bin_img, angle)
    #     scores.append(score)

    # best_score = max(scores)
    # best_angle = angles[scores.index(best_score)]
    # # correct skew
    # data = inter.rotate(bin_img, best_angle, reshape=False, order=0, cval = 1)
    # rotatedImage = 255*data
    # rotated = 255 - rotatedImage
    
    # cv2.imshow('original',threshed)
    # cv2.imshow('rotated',rotated)
    # cv2.waitKey(0)    
    ##################################    
    
    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)
    # print(hist)
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if (y-4>0 and hist[y]==0 and hist[y-1]==0 and hist[y-2]==0 and hist[y-3]==0 and hist[y-4]==0 and hist[y+1]>0)]
    lowers = [y for y in range(H-1) if (y+4<H-1 and hist[y]>0 and hist[y+1]==0 and hist[y+2]==0 and hist[y+3]==0 and hist[y+4]==0)]

    
    # rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    # rotated=255-rotated

    th, threshed = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
    rotated = threshed

    lines = []
    #check we've got the whole page before printing
    for i in range(len(uppers)):
        if uppers[i]>=lowers[i]+1:
            return []

    for y in range(len(uppers)):
        lines += [rotated[uppers[y]:lowers[y]+1,:]]
        if saveResults:
            name="../PreprocessingOutput/LineSegmentation/"+imgName+'-'+"line#"+str(y+1)+".png"
            print(name)
            cv2.imwrite(name,rotated[uppers[y]:lowers[y]+1,:])
    
    
    # if saveResults:
    #     for y in uppers:
    #         cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

    #     for y in lowers:
    #         cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)

    #     cv2.imwrite("result.png", rotated)

    return lines
if __name__ == "__main__":
    
    LineSegmentation(cv2.imread("../Dataset/scanned/caug1200.png"),saveResults = True)