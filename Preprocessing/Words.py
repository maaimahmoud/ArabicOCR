import numpy as np
import cv2

def WordSegmentation(img, imgName = "", lineNumber = 0, saveResults=True):    
    
    # cv2.imshow('original', img)
    # cv2.waitKey(0)

    # bounding_box = cv2.findNonZero(img)
    # x, y, w, h = cv2.boundingRect(bounding_box) # Find minimum spanning bounding box
    # LINE_MARGIN = 1
    # img = img[max(0,y-LINE_MARGIN):min(img.shape[0], y+h+LINE_MARGIN), max(0,x-LINE_MARGIN):min(img.shape[1],x+w+LINE_MARGIN)] 
    
    I = np.asarray(img)

    # Calculate Vertical Projection
    VerticalProjection = cv2.reduce(img, 0, cv2.REDUCE_AVG).reshape(-1)

    # Smoothing Vertical Projection
    avgFilter = np.array([1/4, 1/4, 1/4, 1/4])
    VerticalProjection = np.convolve(VerticalProjection, avgFilter, 'same')

    ####################################
    
    # Gaps Space Locations
    
    Gaps = []
    Lengths = []

    i = 0
    while i < I.shape[1]:
        if VerticalProjection[i] == 0:
            current_length = 0
            while i < I.shape[1] and VerticalProjection[i] == 0:
                current_length += 1
                i += 1
            Gaps += [i]
            Lengths += [current_length]

        i += 1
            
    ####################################

    # Gap Length Filtration
    
    # Calculate IQR
    # n = len(Lengths)/2
    # sortedLengths = Lengths.copy()
    # sortedLengths.sort()

    # Q1 = sortedLengths[int(n/2)]
    # Q2 = sortedLengths[int(n*3/2)]
    # IQR = int((Q1+Q2)/2)
    IQR = 0
    ###############

    filteredGaps = []

    for i in range(len(Gaps)):
        if Lengths[i] > IQR:
            filteredGaps += [Gaps[i]]

    ####################################    
    # Generate Output words after Segmentation

    words = []
    previousGap = I.shape[1]
    del filteredGaps[-1]
    i = 1

    resultImage = np.copy(I)
    resultImage = cv2.cvtColor(resultImage, cv2.COLOR_GRAY2BGR)

    for currentGap in reversed(filteredGaps):
        wordImage = I[:,currentGap:previousGap]
        if np.sum(wordImage) > 0:
            words += [wordImage]
            # if saveResults:
            #     cv2.imwrite("../PreprocessingOutput/WordSegmentation/"+imgName+'-'+"line#"+str(lineNumber)+'-word#'+str(i)+".png",wordImage)
            #     i += 1
        previousGap = currentGap
        cv2.line(resultImage, (currentGap,0), (previousGap, I.shape[1]), (0,255,0), 1)
    # if saveResults:
    #     cv2.imwrite("../PreprocessingOutput/WordSegmentation/"+imgName+'-'+"line#"+str(lineNumber)+".png",resultImage)
    
    return words

if __name__ == "__main__":
    WordSegmentation(cv2.imread('../Dataset/scanned/capr1.png'))