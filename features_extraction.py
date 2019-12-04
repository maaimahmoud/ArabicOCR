import numpy as np
import os
import cv2

def features_extraction(image, black_background):
    features = []                                                     # features vector                     

    height, width, _ = image.shape
    features.append(height / width)

    # get gray and binary version of image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY)

    if black_background:
        binary_img = 1-binary_img

    pixels_count = height * width
    white_pixels = cv2.countNonZero(binary_img)                        # number of white pixels
    black_pixels = pixels_count - white_pixels                         # number of black pixels
    features.append(black_pixels / white_pixels)

    # Calculate vertical and horizontal transitions count
    horizontal_transition_count = 0
    vertical_transition_count = 0
    VerticalLastPixel = np.ones(width)
    lastPixel = 1
    for row in range(height):
            for col in range(width):
                if(lastPixel != binary_img[row,col]):
                    horizontal_transition_count += 1
                lastPixel = binary_img[row,col]
                if(VerticalLastPixel[col] != binary_img[row,col]):
                    vertical_transition_count += 1
                VerticalLastPixel[col] = binary_img[row,col]
    # append vertical and horizontal transitions count to feature vector
    features.append(vertical_transition_count)
    features.append(horizontal_transition_count)

    quarter_pixels = pixels_count / 4
    W1 = cv2.countNonZero(binary_img[0:int(height/2), 0:int(width/2)])           # number of white pixels in first quarter of image
    B1 = quarter_pixels - W1                                                     # number of black pixels in first quarter of image
    W2 = cv2.countNonZero(binary_img[0:int(height/2), int(width/2):width])           
    B2 = quarter_pixels - W2     
    W3 = cv2.countNonZero(binary_img[int(height/2):height, 0:int(width/2)])           
    B3 = quarter_pixels - W3
    W4 = white_pixels - (W1 + W2 + W3)
    B4 = quarter_pixels - W4 

    # avoid division by zero
    if W1 == 0:
        W1 =1 
    if W2 == 0:
        W2 =1 
    if W3 == 0:
        W3 =1 
    if W4 == 0:
        W4 =1 
    if B2 == 0:
        B2 =1 
    if B3 == 0:
        B3 =1 
    if B4 == 0:
        B4 =1 

    features.extend([B1/W1, B2/W2, B3/W3, B4/W4, B1/B2, B1/B3, B1/B4, B2/B3, B2/B4, B3/B4])

    # Determine center of mass (of black ink)
    sx = 0
    sy = 0
    for y in range(height):
        for x in range(width):
            if np.all(binary_img[y,x] == 0):
                sx = sx + x
                sy = sy + y
    if black_pixels == 0:
        black_pixels =1
    centerX = int(sx / black_pixels)
    centerY = int(sy / black_pixels)

    features.extend([centerX/width, centerY/height])

    # Calculate horizontal and vertical histogram
    if not black_background:
        gray = 255-gray
    img_row_sum = np.sum(gray, axis=1).tolist()                                     
    img_col_sum = np.sum(gray, axis=0).tolist()                   
    img_row_sum[:] = [x / width for x in img_row_sum]                                # normalization
    img_col_sum[:] = [x / height for x in img_col_sum]

    img_row_sum_split = np.array_split(img_row_sum, 20)                              # split into 20 chunks
    img_col_sum_split = np.array_split(img_col_sum, 20)

    # append the average of the 20 chunks to feature vector (for horizontal and vertical)
    for chunk in img_col_sum_split:                                                  
        if chunk.size != 0:
            features.append(np.average(chunk))
        else:
            features.append(0)
    for chunk in img_row_sum_split:
        if chunk.size != 0:
            features.append(np.average(chunk))
        else:
            features.append(0)
    return features

img = cv2.imread("Amiri-BoldIsolated_label_4_size_72.png")
features = features_extraction(img, False)
print(features)

#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()