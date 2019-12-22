# -*- coding: utf-8 -*-
import cv2
import numpy as np
from Lines import LineSegmentation
from Words import WordSegmentation
from Characters import CharacterSegmentation

def Preprocssing(img, imgName=""):
    # Segment paragraph into lines    
    lines = LineSegmentation(img, imgName = imgName ,saveResults=True)

    # Segment lines into words
    numberOfExtractedWords = 0
    words = []
    for i in range(len(lines)):
        words += [WordSegmentation(lines[i], imgName = imgName, lineNumber = i + 1 , saveResults=True)]
        numberOfExtractedWords += len(words[-1])

    characters = []
    # Segment words into characters
    # for i in range(len(words)):
    #     for j in range(len(words[i])):
    #             characters += [CharacterSegmentation(np.array(words[i][j], dtype=np.uint8), imgName = imgName, lineNumber= i + 1, wordNumber = j + 1 , saveResults = True)]

    return characters, numberOfExtractedWords

import re
import glob
from tqdm import tqdm
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":

    originalNumberOfWords = 0
    calculatedNumberOfWords = 0

    wrongSegmented = 0
    correctrlySegmented = 0
    
    TRAINING_DATASET = '../Dataset/scanned'

    j = 1
    # Read Image  
    for i in list(sorted(glob.glob(TRAINING_DATASET + "*/*.png"),  key=natural_keys)[:]):
        print(i)
        img = cv2.imread(i)

        textFileName = i[:-4].replace('scanned','text')

        textWords = open(textFileName+'.txt', encoding='utf-8').read().replace('\n',' ').split(' ')

        # textWords = [item for item in textWords if item != '']
        newTextWords = []

        lamAlef = 0

        for item in textWords:

            lamAlef += item.count('لا')
            newTextWords += [item]


        original = len(textWords)
        originalNumberOfWords += original

        __, calculated = Preprocssing(img, imgName = i[ i.rfind('\\') + 1 : -4] )

        j += 1

        calculatedNumberOfWords += calculated

        if original != calculated:
            print(original, calculated, lamAlef)

        if original != calculated:
            wrongSegmented += 1
        else:
            correctrlySegmented += 1
        
        # print("Word Segmentation Accuracy = ",calculated/original*100)

    print("Total Word Segmentation Accuracy = ",calculatedNumberOfWords/originalNumberOfWords*100,' (over ', originalNumberOfWords,' words)')
    
    print("wrongSegmented = ", wrongSegmented, " correctrlySegmented = ", correctrlySegmented)
    
    # Show Paragraph
    
    # cv2.imshow('OriginalImage',img)
    # cv2.waitKey(0)    
    