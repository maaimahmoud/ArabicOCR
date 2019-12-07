import numpy as np
import cv2

def WordSegmentation(img, lineNumber, saveResults=True):    
    
    bounding_box = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(bounding_box) # Find minimum spanning bounding box
    LINE_MARGIN = 1
    cropped = img[max(0,y-LINE_MARGIN):min(img.shape[0], y+h+LINE_MARGIN), max(0,x-LINE_MARGIN):min(img.shape[1],x+w+LINE_MARGIN)] 
    
    I = np.asarray(cropped)

    # Calculate Vertical Projection

    # VerticalProjection = np.zeros(I.shape[1])

    # for col in range(I.shape[1]):
    #     for row in range(I.shape[0]):
    #         VerticalProjection[col] += I[row, col]

    VerticalProjection = cv2.reduce(I, 0, cv2.REDUCE_AVG).reshape(-1)

    old = VerticalProjection.copy()
    # VerticalProjection = cv2.GaussianBlur(VerticalProjection,(4,1),cv2.BORDER_DEFAULT)
    avgFilter = np.array([1/3, 1/3, 1/3])
    VerticalProjection = np.convolve(VerticalProjection, avgFilter, 'same')
    if (old == VerticalProjection).all():
        print("mt8ayrtsh")
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
    n = len(Lengths)/2
    sortedLengths = Lengths.copy()
    sortedLengths.sort()

    Q1 = sortedLengths[int(n/2)]
    Q2 = sortedLengths[int(n*3/2)]

    # IQR = int((Q1+Q2)/2)
    IQR = 1
    ###############

    filteredGaps = []

    for i in range(len(Gaps)):
        if Lengths[i] >= IQR:
            filteredGaps += [Gaps[i]]

    ####################################    
    # Generate Output words after Segmentation

    words = []
    previousGap = I.shape[1]
    del filteredGaps[-1]
    i = 1
    for currentGap in reversed(filteredGaps):
        words += [I[:,currentGap:previousGap]]
        if saveResults:
            cv2.imwrite("../PreprocessingOutput/WordSegmentation/"+str(lineNumber)+'-'+str(i)+".png",I[:,currentGap:previousGap])
            i += 1
        previousGap = currentGap
            
    return words

if __name__ == "__main__":
    WordSegmentation(cv2.imread('../Dataset/scanned/capr1.png'))