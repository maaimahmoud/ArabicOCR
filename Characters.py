import cv2
import numpy as np


def CharacterSegmentation(wordImage):

    ## Gray Scale Conversion
    gray = cv2.cvtColor(wordImage, cv2.COLOR_BGR2GRAY)
    
    # Trimming the non character border parts
    coords = cv2.findNonZero(gray)
    x,y,w,h = cv2.boundingRect(coords)
    word = gray[y:y+h, x:x+w]


    # kernel = np.ones((1.1,1), np.uint8) 
    # word = cv2.erode(trimmed, kernel, iterations=1) 
  

    # Convert to numpy array
    temp = np.asarray(word) 
    I = np.copy(temp)
    cv2.imshow('Word',word)
    cv2.waitKey(0)   

    # Calculating som statistics about the word 
    HorizontalProjection = np.zeros(I.shape[0])
    HorizontalTransition = np.zeros(I.shape[0])
    VerticalProjection = np.zeros(I.shape[1])
    VerticalTransition = np.zeros(I.shape[1])
    VerticalLastPixel = np.zeros(I.shape[1])
    lastPixel = 0

    for row in range(I.shape[0]):
        for col in range(I.shape[1]):
            # Binarizing the image
            if(I[row,col]>127):
                I[row,col] = 1
            else:
                I[row,col] = 0 

            #Horizontal Projection
            HorizontalProjection[row] += I[row, col]
            
            #Vertical Projection
            VerticalProjection[col] += I[row, col]
            
            #Horizontal Transition
            if(lastPixel != I[row,col]):
                HorizontalTransition[row] += 1
            lastPixel = I[row,col]

            #Vertical Transition
            if(VerticalLastPixel[col] != I[row,col]):
                VerticalTransition[col] += 1
            VerticalLastPixel[col] = I[row,col]

    # Getting the most frequent value in the vetical projection lists i.e. the baseline width
    nonzeros = list(filter(lambda a: a != 0, VerticalProjection.astype('uint8')))
    MFV = np.bincount(nonzeros).argmax()
    

    
    word = cv2.cvtColor(word, cv2.COLOR_GRAY2RGB)
    # Getting the baseline index
    maxElement = np.amax(HorizontalProjection)
    index = np.where(HorizontalProjection == maxElement)
    BaselineIndex = index[0][0]
    # cv2.line(word, (0, index[0]), (wordImage.shape[1], BaselineIndex), (0,255,0), 1)
    
    # Getting the index of the maximum horizontal transitions above the baseline
    maxTransitions = np.amax(HorizontalTransition[:BaselineIndex])
    index = np.where(HorizontalTransition == maxTransitions)
    MaxTransitionsIndex = index[0][0]
    # cv2.line(word, (0, MaxTransitionsIndex), (wordImage.shape[1], MaxTransitionsIndex), (255,0,0), 1)

    # Locating te start and end indices of each separation region
    startIndices = []
    endIndices = []
    lastPixel = 0   

    for col in reversed(range(I.shape[1])):
        if(lastPixel == 1 and I[MaxTransitionsIndex,col] == 0):
            startIndices.append(col)
        elif(lastPixel == 0 and I[MaxTransitionsIndex,col] == 1):
            endIndices.append(col)
        lastPixel = I[MaxTransitionsIndex,col]
        
    if(startIndices[0]<endIndices[0]):
        endIndices.pop(0)
    
    if(endIndices[len(endIndices)-1]>startIndices[len(startIndices)-1]):    
        startIndices.pop(-1)
    
    


    # Identifying the cut index for each separation region
    cutIndices = [0]
    
    for i in range(len(startIndices)-1):
        
        start = startIndices[i]
        end = endIndices[i]
        cut = int((start - end) /2) + end
        found = 0 
        
        for col in reversed(range(end, start)):
            if(VerticalProjection[col] == 0):
                found = 1
                cut = col
        
        if(found == 0 and abs(VerticalProjection[cut] - MFV) > 2):
            for col in reversed(range(end,cut)):
                if(abs(VerticalProjection[col] - MFV) < 2):
                    cut = col
                    found = 1
                    break
            if(found == 0):
                for col in range(cut,start):
                    if(abs(VerticalProjection[col] - MFV) < 2):
                        cut = col
                        found = 1
                        break
        if(VerticalTransition[cut]<=2):
            cutIndices.append(cut)
    

    VT = []
    # for s in startIndices:
    #     cv2.circle(word, (s,MaxTransitionsIndex), 1, (255,0,0))
    # for e in endIndices:
    #     cv2.circle(word, (e,MaxTransitionsIndex), 1, (0,255,0))   
    for c in cutIndices:     
        cv2.line(word, (c, 0), (c, wordImage.shape[0]), (0,0,255), 1)
        VT.append(VerticalTransition[c])
    
    print(startIndices)
    print(endIndices)
    print(cutIndices)
    print("MFV ",MFV)
    print(VerticalProjection)
    print(VT)
    
    # plot image
    cv2.imshow('Word',word)
    cv2.waitKey(0)    




    # To Save output image
    cv2.imwrite("PreprocessingOutput/CharacterSegmentation/Char1.png",word)
    pass


if __name__ == "__main__":
    CharacterSegmentation(cv2.imread("PreprocessingOutput/WordSegmentation/9-5.png"))