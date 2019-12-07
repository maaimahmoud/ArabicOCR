import cv2
import numpy as np
from Lines import LineSegmentation
from Words import WordSegmentation
from Characters import CharacterSegmentation

def Preprocssing(img):
    # Segment paragraph into lines    
    lines = LineSegmentation(img, saveResults=False)

    # Segment lines into words
    words = []
    for i in range(len(lines)):
        words += [WordSegmentation(lines[i], i, saveResults=True)]

    # print(words[0,0].shape)
    characters = []
    # Segment words into characters
    for i in range(len(words)):
        for j in range(len(words[i])):
                print(i, j)
                characters += [CharacterSegmentation(cv2.cvtColor(np.array(words[i][j], dtype=np.uint8), cv2.COLOR_GRAY2BGR), lineNumber=i, wordNumber =j)]

    return characters


if __name__ == "__main__":
    # Read Image    
    img = cv2.imread("../Dataset/scanned/capr6.png")

    # Show Paragraph
    # cv2.imshow('OriginalImage',img)
    # cv2.waitKey(0)    

    Preprocssing(img)