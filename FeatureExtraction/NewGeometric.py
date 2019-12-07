import cv2
import numpy as np
import collections

class NewGeometric():

    def getFeatures(self, gray):
        features = []

        # 1) Convert the input image to a binary image with 0 and 1.

        ################# fill holes #################
        # 2) Remove black holes with values 0 that surround with 1.

        th, im_th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)    
        # Copy the thresholded image.
        im_floodfill = im_th.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
        gray = im_out
        cv2.imshow('RemoveHoles',gray)
        cv2.moveWindow('RemoveHoles', 100,100)
        # cv2.imshow("Thresholded Image", im_th)
        # cv2.imshow("Floodfilled Image", im_floodfill)
        # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        # cv2.imshow("Foreground", im_out)    

        ################# Remove black spaces #################
        # gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
        bounding_box = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(bounding_box) # Find minimum spanning bounding box
        CHAR_MARGIN = 1
        cropped = gray[max(0,y-CHAR_MARGIN):min(gray.shape[0], y+h+CHAR_MARGIN), max(0,x-CHAR_MARGIN):min(gray.shape[1],x+w+CHAR_MARGIN)] # 0, 255
        # I = np.asarray(cropped)
        cv2.imshow('RemoveSpaces',cropped)
        cv2.moveWindow('RemoveSpaces', 300,100)
        ################# Resize Image #################
        # 3) Resize of the input image to 100 x 60 pixels.
        resized = cv2.resize(cropped, (60, 100)) # 7, 11 , 10 ,255, 0
        th, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY) # 0, 255
        cv2.imshow('ResizeImage',resized)
        cv2.moveWindow('ResizeImage', 500,100)
        ################# Divide to 4 regions #################
        # 4) Find the centre of the resized image (cx, cy).
        current_image = resized.copy()
        center = (int(current_image.shape[0]/2),int(current_image.shape[1]/2))

        # 5) Calculate the number of pixels with value 1.
        current_image = current_image / 255 # 0, 1 
        unique, counts = np.unique(current_image, return_counts=True)
        NumberOfPixelsWithValue1 = dict(zip(unique, counts))[1]
        features += [NumberOfPixelsWithValue1] # F1

        # 6) Four parts (C1:4) divided image and the area is calculate for all parts.
        left  = 0                     , center[0]
        right = current_image.shape[1], center[0]
        up    = center[1]             ,  0
        down  = center[1]             , current_image.shape[0]
        
        # # Check that division is correct
        # C1Area = abs(up[0]   - right[0]) * abs(up[1]   - right[1])
        # C2Area = abs(up[0]   - left[0])  * abs(up[1]   - left[1])
        # C3Area = abs(down[0] - right[0]) * abs(down[1] - right[1])
        # C4Area = abs(down[0] - left[0])  * abs(down[1] - left[1])
        # print("Total = ",current_image.shape[0]*current_image.shape[1]," calculated = ", C1Area+C2Area+C3Area+C4Area)

        C1Image = current_image[:center[0], center[1]:]
        unique, counts = np.unique(C1Image, return_counts=True)
        myDict = dict(zip(unique, counts))
        if (1 not in myDict):
            myDict[1] = 0
        C1 = myDict[1]
        cv2.imshow('C1', C1Image)
        cv2.moveWindow('C1', 100,700)
        
        C2Image = current_image[:center[0],:center[1]]
        unique, counts = np.unique(C2Image, return_counts=True)
        myDict = dict(zip(unique, counts))
        if (1 not in myDict):
            myDict[1] = 0
        C2 = myDict[1]
        cv2.imshow('C2', C2Image)
        cv2.moveWindow('C2', 300,700)

        C3Image = current_image[ center[0]: , center[1]: ]
        unique, counts = np.unique(C3Image, return_counts=True)
        myDict = dict(zip(unique, counts))
        if (1 not in myDict):
            myDict[1] = 0
        C3 = myDict[1]
        cv2.imshow('C3',C3Image)
        cv2.moveWindow('C3', 500,700)
        
        C4Image = current_image[ center[0]: , :center[1]]
        unique, counts = np.unique(C4Image, return_counts=True)
        myDict = dict(zip(unique, counts))
        if (1 not in myDict):
            myDict[1] = 0
        C4 = myDict[1]
        cv2.imshow('C4', C4Image)
        cv2.moveWindow('C4', 700,700)

        features += [C1, C2, C3, C4] #F2, F3, F4, F5


        fourRegionsOutput = current_image.copy() # 0, 1
        cv2.line(current_image, left, right, (255,255,255), 2)
        cv2.line(current_image, up, down, (255,255,255), 2)
        cv2.imshow('C1:C4',current_image)
        cv2.moveWindow('C1:C4', 700,100)


        ################# Edge Detection #################
        # 7) Apply Soble edge detection method.
        current_image = fourRegionsOutput.copy()
        sobelx = cv2.Sobel(current_image, cv2.CV_8U, dx=0, dy=1, ksize=5)
        sobely = cv2.Sobel(current_image, cv2.CV_8U, dx=1, dy=0, ksize=5)
        cv2.imshow('SobelEdgeDetection',sobelx+sobely)
        cv2.moveWindow('SobelEdgeDetection', 900,100)
        
        laplacian = cv2.Laplacian(current_image,cv2.CV_64F)
        cv2.imshow('LaplacianEdgeDetection', laplacian)
        cv2.moveWindow('LaplacianEdgeDetection', 1100,100)

        # 8) Four parts (E1:4) divided image edges and the area is calculate for all part.
        current_image = laplacian.copy()
        current_image[current_image != 0 ] = 1
        # E1Area = abs(up[0] - right[0]) * abs(up[1] - right[1])
        # E2Area = abs(up[0] - left[0]) * abs(up[1] - left[1])
        # E3Area = abs(down[0] - right[0]) * abs(down[1] - right[1])
        # E4Area = abs(down[0] - left[0]) * abs(down[1] - left[1])
        


        E1Image = current_image[:center[0], center[1]:]
        unique, counts = np.unique(E1Image, return_counts=True)
        myDict = dict(zip(unique, counts))
        if (1 not in myDict):
            myDict[1] = 0
        E1 = myDict[1]
        cv2.imshow('E1', E1Image)
        cv2.moveWindow('E1', 100,800)
        
        E2Image = current_image[:center[0],:center[1]]
        unique, counts = np.unique(E2Image, return_counts=True)
        myDict = dict(zip(unique, counts))
        if (1 not in myDict):
            myDict[1] = 0
        E2 = myDict[1]
        cv2.imshow('E2', E2Image)
        cv2.moveWindow('E2', 300,800)

        E3Image = current_image[ center[0]: , center[1]: ]
        unique, counts = np.unique(E3Image, return_counts=True)
        myDict = dict(zip(unique, counts))
        if (1 not in myDict):
            myDict[1] = 0
        E3 = myDict[1]
        cv2.imshow('E3',E3Image)
        cv2.moveWindow('E3', 500,800)
        
        E4Image = current_image[ center[0]: , :center[1]]
        unique, counts = np.unique(E4Image, return_counts=True)
        myDict = dict(zip(unique, counts))
        if (1 not in myDict):
            myDict[1] = 0
        E4 = myDict[1]
        cv2.imshow('E4', E4Image)
        cv2.moveWindow('E4', 700,800)

        features += [E1, E2, E3, E4] #F6, F7, F8, F9

        edgeDetectionOutput = current_image.copy()

        cv2.line(current_image, left, right, (255,255,255), 2)
        cv2.line(current_image, up, down, (255,255,255), 2)
        cv2.imshow('E1:E4',current_image)
        cv2.moveWindow('E1:E4', 1300,100)

        ################# Bounding Box of Character #################
        # 9) Find the first pixel of 1 in the first row, last column, last row and first column, (x1, y1), (x2, y2), (x3,
        # y3), and (x4, y4), respectively. Connect these points to get four lines L1, L2, L3, and L4 then calculate
        # the length of each line.

        current_image = edgeDetectionOutput.copy()
        
        transposeMat = cv2.transpose(current_image)

        rowIndex = np.where(current_image == 1)

        colIndex = np.where(transposeMat == 1)
        
        # print(colIndex)
        
        temp = zip(rowIndex[0], rowIndex[1])
        rowOnes = tuple(temp)

        temp = zip(colIndex[0], colIndex[1])
        colOnes = tuple(temp)
        
        # x1, y1 ==> pixel in first row
        y1, x1 = rowOnes[0]

        # x2, y2 => pixel in last column
        x2, y2 = colOnes[-1]

        # x3, y3 ==> pixel in last row
        y3, x3 = rowOnes[-1]

        # x4, y4 => pixel in first column
        x4, y4 = colOnes[0]
        
        boundingBox = current_image.copy()
        
        # print('(',x1, y1,')','(', x2, y2,')', '(',x3,y3,')','(',x4, y4,')')
        cv2.line(current_image, (x1, y1 ), (x2, y2 ), (255,255,255), 2)
        cv2.line(current_image, (x2, y2 ), (x3, y3 ), (255,255,255), 2)
        cv2.line(current_image, (x3, y3 ), (x4, y4 ), (255,255,255), 2)
        cv2.line(current_image, (x4, y4 ), (x1, y1 ), (255,255,255), 2)
        cv2.imshow('Bounding Box of characters',current_image)
        cv2.moveWindow('Bounding Box of characters', 1500,100)

        import math
        def lineLength(point1, point2):
            return math.sqrt(pow( ( point1[0]-point2[0] ) , 2)+ pow( ( point1[1]-point2[1] ), 2))

        L1 = lineLength( (x1, y1), (x2, y2 ))
        L2 = lineLength( (x2, y2), (x3, y3 ))
        L3 = lineLength( (x3, y3), (x4, y4 )) 
        L4 = lineLength( (x4, y4), (x4, y4 )) 

        features += [L1, L2, L3, L4] # F10, F11, F12, F13
        #######################################################
        # 10) For all rows find the first and last pixel that equal to 1. Plot straight line by connecting these two points.
        
        current_image = fourRegionsOutput.copy()
        # horizontalLines = np.asarray(current_image)
        continousHorizontalLines = current_image.copy()
        continousHorizontalLines[:,:] = 0

        discreteHorizontalLines = current_image.copy()
        discreteHorizontalLines[:,:] = 0

        discreteRows = 0
        continousRows = 0
        totalRows = 0

        # Longest
        T1 = -1
        T1pos = 0,0

        # shortest
        T2 = 200
        T2pos = 0,0


        idx = 0
        for i in range(current_image.shape[0]):
            change = 0
            for j in range(0, current_image.shape[1]-1):
                if  current_image[i,j] != current_image[i,j+1]:
                    change += 1
            if change == 2:
                continousRows += 1
                ones = np.where(current_image[i] == 1)
                cv2.line(continousHorizontalLines, (ones[0][0], idx), ( ones[0][-1], idx), (255,255,255), 2)
                if (len(ones[0]) > T1):
                    T1 = len(ones[0])
                    T1pos = ones[0][0], idx
                    
                if (len(ones[0]) < T2):
                    T2 = len(ones[0])
                    T2pos = ones[0][0], idx

            elif change > 0:
                discreteRows += 1
                ones = np.where(current_image[i] == 1)
                cv2.line(discreteHorizontalLines, (ones[0][0], idx), ( ones[0][-1], idx), (255,255,255), 2)

            idx += 1


        cv2.imshow('Discrete Horizontal Lines',discreteHorizontalLines)
        cv2.moveWindow('Discrete Horizontal Lines', 100,400)

        
        cv2.imshow('Continous Horizontal Lines',continousHorizontalLines)
        cv2.moveWindow('Continous Horizontal Lines', 300,400)

        # 11) Account the number of lines that do not have zeroes in the middle, and those that have zeroes in its
        # middle.

        # 12) Find the longest and shortest straight line without zeroes in middle and its location in every row namely
        # T1, T2, Tpo1, and Tpo2 respectively.

        features += [T1, T1pos[0], T1pos[1], T2, T2pos[0], T2pos[1]] #F14, F15, F16, F17, F18, F19
        # print(features)
        # cv2.waitKey(0)

        return features

if __name__ == "__main__":
    
    # Read Image    
    img = cv2.imread("../Dataset/daal.png")

    # Gray Scale Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('OriginalImage',gray)
    cv2.moveWindow('OriginalImage', 100,100)
    # cv2.waitKey(0)    

    newGeo = NewGeometric()
    newGeo.getFeatures(gray)