import cv2
import numpy as np

def CharacterSegmentation(wordImage):

    ## Gray Scale Conversion
    gray = cv2.cvtColor(wordImage, cv2.COLOR_BGR2GRAY)

    # Convert to numpy array
    I = np.asarray(gray) 

    # plot image
    cv2.imshow('Word',I)
    cv2.waitKey(0)    


    ##############################



    ##############################


    # To Save output image
    cv2.imwrite("PreprocessingOutput/CharacterSegmentation/Char1.png",I)
    pass


if __name__ == "__main__":
    CharacterSegmentation(cv2.imread("PreprocessingOutput/WordSegmentation/1-4.png"))