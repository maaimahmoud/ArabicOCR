import cv2
import numpy as np
from Lines import LineSegmentation
from Words import WordSegmentation
from Characters import CharacterSegmentation

def Preprocssing(img):
    # Segment paragraph into lines    
    lines = LineSegmentation(img, saveResults=True)

    # Segment lines into words
    numberOfExtractedWords = 0
    words = []
    for i in range(len(lines)):
        words += [WordSegmentation(lines[i], i, saveResults=True)]
        numberOfExtractedWords += len(words[-1])

    # print(words[0,0].shape)
    characters = []
    # Segment words into characters
    # for i in range(len(words)):
    #     for j in range(len(words[i])):
    #             # print(i, j)
    #             characters += [CharacterSegmentation(cv2.cvtColor(np.array(words[i][j], dtype=np.uint8), cv2.COLOR_GRAY2BGR), lineNumber=i, wordNumber =j)]

    return characters, numberOfExtractedWords


if __name__ == "__main__":

    originalNumberOfWords = 0
    calculatedNumberOfWords = 0

    # Read Image    
    for i in range(6,7):
        img = cv2.imread("../Dataset/scanned/capr"+str(i)+".png")
        text = open("../Dataset/text/capr"+str(i)+".txt",encoding='utf-8')
        # print(text.read())

        original = len(text.read().replace('\n','').split(' '))
        originalNumberOfWords += original

        __, calculated = Preprocssing(img)
        calculatedNumberOfWords += calculated

        print(original, calculated)
        
        print("Word Segmentation Accuracy = ",calculated/original*100)

    print("Total Word Segmentation Accuracy = ",calculatedNumberOfWords/originalNumberOfWords*100,' (over ', originalNumberOfWords,' words)')
    # Show Paragraph
    # cv2.imshow('OriginalImage',img)
    # cv2.waitKey(0)    
    