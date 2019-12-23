# -*- coding: utf-8 -*-
import cv2
import numpy as np
from Lines import LineSegmentation
from Words import WordSegmentation
from Characters import CharacterSegmentation
import timeit

def Preprocssing(img, imgName="", textWords = 0):
    # Segment paragraph into lines    
    lines = LineSegmentation(img, imgName = imgName ,saveResults=False)

    # Segment lines into words
    # numberOfExtractedWords = 0
    words = []
    for i in range(len(lines)):
        words.extend(WordSegmentation(lines[i], imgName = imgName, lineNumber = i + 1 , saveResults=False))
        # numberOfExtractedWords += len(words[-1])

    if len(textWords) != len(words):
        # print("ERROR IN WORD SEGMENTATION IN ", imgName)

    # import time
    # time.sleep(10)

        return [], 0

    characters = []
    # Segment words into characters
    # for i in range(len(words)):
    #     for j in range(len(words[i])):
    #             characters += [CharacterSegmentation(np.array(words[i][j], dtype=np.uint8), imgName = imgName, lineNumber= i + 1, wordNumber = j + 1 , saveResults = True)]
    i = 0
    charError = 0
    for word in words:
        characters.append(CharacterSegmentation(np.array(word, dtype=np.uint8), imgName = imgName, lineNumber= i + 1, wordNumber = np.ceil((i + 1)/len(lines) ), saveResults = True))
        if len(textWords[i])-textWords[i].count('لا') != len(characters[-1]):
            # print("ERROR IN CHARACTER SEGMENTATION IN ", imgName," Word #", i)
            charError += 1
        i += 1
        print(i , len(characters[-1]))
    return characters, charError

import re
import glob
from tqdm import tqdm
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":

    originalNumberOfWords = 0
    calculatedNumberOfWords = 0

    wrongSegmented = 0
    correctrlySegmented = 0

    totalcharError = 0
    totalWords = 0

    TRAINING_DATASET = '../Dataset/scanned'

    j = 1
    # Read Image
    start_time = timeit.default_timer()
    for i in list(sorted(glob.glob(TRAINING_DATASET + "*/*.png"),  key=natural_keys)[2:3]):
        # print(i)
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


        __, charError = Preprocssing(img, imgName = i[ i.rfind('\\') + 1 : -4] , textWords=textWords)

        if charError != 0:
            totalWords += len(textWords)
        j += 1

        totalcharError += charError

        # calculatedNumberOfWords += calculated
        #
        # if original != calculated:
        #     print(original, calculated, lamAlef)
        #
        # if original != calculated:
        #     wrongSegmented += 1
        # else:
        #     correctrlySegmented += 1
        #
        # print("Word Segmentation Accuracy = ",calculated/original*100)
    print("Runtime:", timeit.default_timer() - start_time)

    # print("Total Word Segmentation Accuracy = ",calculatedNumberOfWords/originalNumberOfWords*100,' (over ', originalNumberOfWords,' words)')
    
    # print("wrongSegmented = ", wrongSegmented, " correctrlySegmented = ", correctrlySegmented)

    print("CHAR SEGMENTATION: WRONG = ", totalcharError, " out of ", totalWords)
    # Show Paragraph
    
    # cv2.imshow('OriginalImage',img)
    # cv2.waitKey(0)    
    