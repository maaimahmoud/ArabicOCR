import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import time
import math


avgCharCount=4
avgLetterWidth=10

def CharacterSegmentation1(gray, imgName = "0", lineNumber = 0, wordNumber = 0, saveResults = False):
    # avg issss 4 -5 words
    coords = cv2.findNonZero(gray)
    x,y,w,h = cv2.boundingRect(coords)
    word = gray[y:y+h, x:x+w]

    # print("word shaape", word.shape) # 0 is h and 1 is w
    avgWidth=math.ceil(word.shape[1]/avgCharCount)
    Characters = []	
    for i in range(avgCharCount):     	
        character = word[:,i*int(avgWidth):min((i+1)*int(avgWidth),word.shape[1])]    	

        # cv2.imwrite("../PreprocessingOutput/CharSegmentation/"+str(imgName)+str(i)+".jpg",character)    
        if character.shape[1]>0:
            Characters.append(character)	

    return Characters	


def CharacterSegmentation(gray, imgName = "0", lineNumber = 0, wordNumber = 0, saveResults = False):
    # avg issss 4 -5 words
    coords = cv2.findNonZero(gray)
    x,y,w,h = cv2.boundingRect(coords)
    word = gray[y:y+h, x:x+w]

    # print("word shaape", word.shape) # 0 is h and 1 is w
    charCount=math.ceil(word.shape[1]/avgLetterWidth)
    avgWidth=math.ceil(word.shape[1]/charCount)
    Characters = []	
    for i in range(charCount):     	
        character = word[:,i*int(avgWidth):min((i+1)*int(avgWidth),word.shape[1])]    	

        # cv2.imwrite("../PreprocessingOutput/CharSegmentation/"+str(imgName)+str(i)+".jpg",character)    
        if character.shape[1]>0:
            Characters.append(character)	

    return Characters	





if __name__ == "__main__":
   
    testCase = sys.argv[1]
    img = cv2.imread("../PreprocessingOutput/WordSegmentation/"+testCase+".jpg")
    
    
    
    # cv2.imshow('original', img)
    # cv2.waitKey(0)

    print(len(CharacterSegmentation(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), saveResults=True)))
      