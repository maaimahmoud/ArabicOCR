import cv2
import numpy as np

def CharacterSegmentation(wordImage):

    ## Gray Scale Conversion
    gray = cv2.cvtColor(wordImage, cv2.COLOR_BGR2GRAY)

    # Convert to numpy array
    I = np.asarray(gray) 


    # Getting the baseline index and maximum transitions index
    HorizontalProjection = np.zeros(I.shape[0])
    HorizontalTransition = np.zeros(I.shape[0])

    lastPixel = 0
    for row in range(I.shape[0]):
        for col in range(I.shape[1]):
            HorizontalProjection[row] += I[row, col]
            if(lastPixel != I[row,col]):
                HorizontalTransition[row] += 1
            lastPixel = I[row,col]


    maxElement = np.amax(HorizontalProjection)
    index = np.where(HorizontalProjection == maxElement)
    cv2.line(wordImage, (0, index[0]), (wordImage.shape[1], index[0]), (0,255,0), 1)

    # TODO: start the array of transitions from the baseline index
    maxTransitions = np.amax(HorizontalTransition)
    index = np.where(HorizontalTransition == maxTransitions)
    cv2.line(wordImage, (0, index[0]), (wordImage.shape[1], index[0]), (255,0,0), 1)

    # plot image
    cv2.imshow('Word',wordImage)
    cv2.waitKey(0)    


    ##############################



    ##############################


    # To Save output image
    cv2.imwrite("PreprocessingOutput/CharacterSegmentation/Char1.png",wordImage)
    pass


if __name__ == "__main__":
    CharacterSegmentation(cv2.imread("PreprocessingOutput/WordSegmentation/7-7.png"))