import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import time
 
def neighbours(x, y, image):
    i = image
    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9
 
def transitions(neighbours):
    n = neighbours + neighbours[0:1] 
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))
 
def skeletonize(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1 
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P4 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P6 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing1.append((x,y))
        for x, y in changing1: image[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P2 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing2.append((x,y))
        for x, y in changing2: image[y][x] = 0
    return image
    

def imageAnalysis(imgArr, xstart, xend, ystart, yend):

    # X higher value is at right
    xportion = xstart - xend
    
    # Y higher value is at down 
    yportion = ystart - yend
    
    HP = np.zeros(imgArr.shape[0], dtype=int)
    HT = np.zeros(imgArr.shape[0], dtype=int)
    VP = np.zeros(imgArr.shape[1], dtype=int)
    VT = np.zeros(imgArr.shape[1], dtype=int)
    VLP = np.zeros(imgArr.shape[1], dtype=int)
    Heights = np.zeros(imgArr.shape[1], dtype=int)
    Heights[:] = -1
    LP = 0
    for row in range(yend, ystart):
        for col in range(xend, xstart):
            
            # Height Calculation
            if(Heights[col] == -1 and imgArr[row, col] == 1):
                Heights[col] = row

            #Horizontal Projection
            HP[row] += imgArr[row, col]
            
            #Vertical Projection
            VP[col] += imgArr[row, col]
            
            #Horizontal Transition
            if(LP != imgArr[row,col]):
                HT[row] += 1
            LP = imgArr[row,col]

            #Vertical Transition
            if(VLP[col] != imgArr[row,col]):
                VT[col] += 1
            VLP[col] = imgArr[row,col]
    
    return HP, HT, VP, VT, Heights


def CharacterSegmentation(wordImage, lineNumber=0, wordNumber =0):

    # I M A G E  P R E P R O C E S S I N G :
    # ---------------------------------------

    ## Gray Scale Conversion
    gray = cv2.cvtColor(wordImage, cv2.COLOR_BGR2GRAY)
    
    # Trimming the non character border parts
    coords = cv2.findNonZero(gray)
    x,y,w,h = cv2.boundingRect(coords)
    word = gray[y:y+h, x:x+w]

    wordLines = gray[y:y+h, x:x+w]
    wordDots = gray[y:y+h, x:x+w]
  
    # Binarizing the image
    ret, binarized = cv2.threshold(word,115,255,cv2.THRESH_BINARY)
    binarized = binarized/255

    # Convert to numpy array
    temp = np.asarray(binarized) 
    I = np.copy(temp)
    nonSkeleton = np.copy(temp)


    # -------------------------------------------------------------------------------------------------
    # I M A G E  S T A T I S T I C S :
    # --------------------------------
  
    # Calculating som statistics about the word 
    HorizontalProjection, HorizontalTransition, VerticalProjection, VerticalTransition, Height = imageAnalysis(I, I.shape[1], 0, I.shape[0], 0)
    
    # Getting the most frequent value in the vetical projection lists i.e. the baseline width
    nonzeros = list(filter(lambda a: a != 0, VerticalProjection.astype('uint8')))
    MFV = np.bincount(nonzeros).argmax()
    
    
    # Getting the baseline index
    maxElement = np.amax(HorizontalProjection)
    index = np.where(HorizontalProjection == maxElement)
    BaselineIndex = index[0][0]
    
    # Getting the index of the maximum horizontal transitions above the baseline
    maxTransitions = np.amax(HorizontalTransition[:BaselineIndex])
    index = np.where(HorizontalTransition == maxTransitions)
    MaxTransitionsIndex = index[0][0]
    if(BaselineIndex - MaxTransitionsIndex > 3):
        MaxTransitionsIndex = BaselineIndex - 3

    # Getting heighest pixel
    Height = [BaselineIndex - Height[index] if Height[index] != -1 else Height[index] for index in range(len(Height))]
    Heighest = max(Height)


    # -------------------------------------------------------------------------------------------------
    # S E P A R A T I O N   R E G I O N S :
    # -------------------------------------
    
    # Locating te start and end indices of each separation region
    startIndices = []
    endIndices = []
    lastPixel = 0   

    for col in reversed(range(I.shape[1])):
        if(lastPixel == 1 and I[MaxTransitionsIndex,col] == 0):
            startIndices.append(col +1)
        elif(lastPixel == 0 and I[MaxTransitionsIndex,col] == 1):
            endIndices.append(col)
        lastPixel = I[MaxTransitionsIndex,col]

    if(startIndices[0] - 2 < endIndices[0]):
        endIndices.pop(0)
    
    if(endIndices[-1] > startIndices[-1] - 2):    
        startIndices.pop(-1)


    # -------------------------------------------------------------------------------------------------
    # C U T   I N D I C E S :
    # -----------------------

    # Skeletonizing the image
    ret, binarized = cv2.threshold(word,115,255,cv2.THRESH_BINARY)
    binarized = binarized/255
    skeletonized = skeletonize(binarized)

    # Convert to numpy array
    temp = np.asarray(skeletonized) 
    I = np.copy(temp)

    # Getting some skeltonized statistics
    HorizontalProjection, HorizontalTransition, VerticalProjection, VerticalTransition, dummy = imageAnalysis(I, I.shape[1], 0, I.shape[0], 0)

    MFV = 1


    # Identifying the cut index for each separation region

    # V A R I A B L E S
    #------------------

    cutIndices = [(I.shape[1]-1,0)]
    Strokes = np.zeros(len(startIndices) + 2)
    currentStroke = 0
    previousStroke = 0
    beforepreviousStroke = 0
    previousHole = 0
    currentBlack = 0
    previousBlack = 0 
    miniStroke = 0
    
    
    # L O O P
    #--------    
    
    for i in range(len(startIndices)):
    
    
        # V A R I A B L E S
        #------------------

        start = startIndices[i]
        end = endIndices[i]
        cut = int((start - end) /2) + end
        found = 0    
        currentStroke = 0

        HPa, HTa, VPa, VTa, q = imageAnalysis(nonSkeleton, start, end+1, BaselineIndex, 0) 
        HPb, HTb, VPb, VTb, q = imageAnalysis(nonSkeleton, start, end+1, I.shape[0], BaselineIndex) 
        
        HPasum = 0 
        for hpa in HPa: 
            HPasum += hpa
            
        HPbsum = 0
        for hpb in HPb:
            HPbsum += hpb


        # C U T  I D E N T I F I C A T I O N
        #-----------------------------------

        # Cut at Subwords in the same word, Remove cuts at false strokes
        for col in reversed(range(end+1, start)):
            if(VerticalProjection[col] == 0):
                found = 1
                cut = col
        if(found == 1):
            if(miniStroke == 1 and HPbsum == 0 and previousBlack == 0):
                cutIndices.pop(-1)
                miniStroke = 0    
            cutIndices.append((cut,0))
            previousBlack = 1
            continue
        
        previousBlack = 0

        # Ba', Ta' cases in the last region
        if(i == len(startIndices) - 1 and start - end > 6):  
            break

        # Detect false strokes at Dal, Qaf, Sad in last region    
        miniStroke = 0    
        if(2*Height[end-1] < Heighest and 2*Height[end] < Heighest and 2*Height[end+1] < Heighest):
            miniStroke = 1          


        # Cut at Baseline only 
        if(abs(VerticalProjection[cut] - MFV)<2 and VerticalTransition[cut] == 2):
            found = 1


        # Searching for Baseline only inside the separation region
        if(found == 0 and VerticalTransition[cut] > 2):
            for col in reversed(range(end+1,cut)):
                if((VerticalProjection[cut] - MFV)<2 and VerticalTransition[col] == 2):
                    cut = col
                    found = 1
                    break
            if(found == 0):
                for col in range(cut+1,start):
                    if((VerticalProjection[cut] - MFV)<2 and VerticalTransition[col] == 2):
                        cut = col
                        found = 1
                        break

        # Adding the cut index unless it's a hole letter
        if(abs(VerticalProjection[cut] - MFV) < 2 and VerticalTransition[cut] == 2):
            cutIndices.append((cut,0))
        else:
            miniStroke = 0

        # No need for further checks
        if(len(cutIndices)==1):
            continue  


        # Detect false segmentation at qaf, sen ...
        if(HPbsum > HPasum and i == len(startIndices)-1 and max(Height[end-1:end+1] if Height[end-1:end+1] != [] else [1])<5):  # check also that the character is not Alif
            cutIndices.pop(-1) 
            continue


        # S T R O K E   D E T E C T I O N
        #--------------------------------

        # Segment Analysis
        HPa, HTa, q, q, q = imageAnalysis(nonSkeleton, cutIndices[-2][0], cut+1, BaselineIndex, 0)
        HPb, HTb, q, q, q = imageAnalysis(nonSkeleton, cutIndices[-2][0], cut+1, I.shape[0], BaselineIndex + 3)
        q, q, VP, VT, q = imageAnalysis(I, cutIndices[-2][0], cut, I.shape[0], 0) 
        HPasum = 0 
        for hpa in HPa[0:int(BaselineIndex/3)]:
            HPasum += hpa
            
        HPbsum = 0
        for hpb in HPb:
            HPbsum += hpb

        # Width of Stroke
        nonzeros = list(filter(lambda a: a != 0, HPa.astype('uint8')))
        if(len(nonzeros)==0): #Check stroke removed during skeletonizing
            nonzeros.append(1)
            HPasum += 1
        MFHP = np.bincount(nonzeros).argmax()

        # Height of he stroke
        height = max(Height[cut:cutIndices[-2][0]])

        # Stroke Detector
        if(HPasum == 0 and HPbsum == 0 and height <= 6 and abs(MFHP - MFV) < 3 and (np.count_nonzero(VT > 2) < 3) and abs(VerticalProjection[cut] - MFV) < 2 and VerticalTransition[cut] == 2):
            currentStroke = 1
            cutIndices[-1] = (cut, 1)




    # S T R O K E   F I L T R A T I O N 
    #----------------------------------

    
    i = 0
    c=0
    while(i<len(cutIndices)):

        c+=1
        if(c==15):
            break
       
        if(cutIndices[i][1] == 1):
           
            if(i < len(cutIndices) - 2):
                if(cutIndices[i+2][1]==1):
                    cutIndices.pop(i)
                    cutIndices.pop(i)
                    cutIndices[i]=(cutIndices[i][0],0)
                else:
                    cutIndices.pop(i-1)
           
            elif(i < len(cutIndices) - 1 and cutIndices[i+1][1]==1):
                cutIndices.pop(i)
                cutIndices.pop(i)
           
            else:
                cutIndices.pop(i-1)
        
        else:
            i+=1
    


    # -------------------------------------------------------------------------------------------------
    # O U T P U T :
    # -------------

    word = cv2.cvtColor(word, cv2.COLOR_GRAY2RGB)
    wordDots = cv2.cvtColor(wordDots, cv2.COLOR_GRAY2RGB)
    wordLines = cv2.cvtColor(wordLines, cv2.COLOR_GRAY2RGB)
    skeletonized *= 255

    VT = []
    for i in range(len(startIndices)):
        wordDots[MaxTransitionsIndex+(i%2),startIndices[i]]= [255*(i%2), 0,255*((i+1)%2)]
        wordDots[MaxTransitionsIndex+(i%2),endIndices[i]]= [255*(i%2), 0,255*((i+1)%2)]
    for c, strock in cutIndices:     
        cv2.line(wordLines, (c, 0), (c, wordImage.shape[0]), (0,0,255), 1)
        VT.append(VerticalTransition[c])

    # To Save output image
    cv2.imwrite("../PreprocessingOutput/CharacterSegmentation/"+str(lineNumber)+'-'+str(wordNumber)+'-'+"Lines.png",wordLines)
    cv2.imwrite("../PreprocessingOutput/CharacterSegmentation/"+str(lineNumber)+'-'+str(wordNumber)+'-'+"Dots.png",wordDots)
    cv2.imwrite("../PreprocessingOutput/CharacterSegmentation/"+str(lineNumber)+'-'+str(wordNumber)+'-'+"skeleton.png",skeletonized)
    cv2.imwrite("../PreprocessingOutput/CharacterSegmentation/"+str(lineNumber)+'-'+str(wordNumber)+'-'+"original.png",wordImage)
    pass


# E N D  O F  C H A R A C T E R  S E G M E N T A T I O N 
# -------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------
# M A I N :
# ---------

if __name__ == "__main__":
   
    testCase = sys.argv[1]
    img = cv2.imread("../PreprocessingOutput/WordSegmentation/"+testCase+".png")
    
    # cv2.imshow('original', img)
    # cv2.waitKey(0)

    CharacterSegmentation(img)
      