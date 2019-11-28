import cv2
from lines import LineSegmentation
from Words import WordSegmentation
from Characters import CharacterSegmentation


if __name__ == "__main__":
    # Read Image    
    img = cv2.imread("Dataset/scanned/capr3.png")

    # Show Paragraph
    cv2.imshow('OriginalImage',img)
    cv2.waitKey(0)    

    # Segment paragraph into lines    
    lines = LineSegmentation(img, saveResults=True)

    # Segment lines into words
    words = []
    for i in range(len(lines)):
        words += [WordSegmentation(lines[i], i, saveResults=True)]


    # Segment words into characters
    # for i in range(len(words)):
    #     CharacterSegmentation(words[i])