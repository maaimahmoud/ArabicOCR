import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
try:
    from tensorflow.contrib import learn
except ImportError:
    from tensorflow import estimator as learn
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import joblib
import pickle

# def read_images_in_folder(folder):
#     images = []
#     folders = [f for f in glob.glob(folder + "**/*", recursive=True)]
#     for folder in folders:
#         img_files = [img_file for img_file in glob.glob(folder + "**/*.png", recursive=True)]
#         for i in range(len(img_files)):
#             img = cv2.imread(img_files[i])    
#             images.append(img)
#             label = int(img_files[i][img_files[i].index("label_") + len("label_"):img_files[i].index("_size")])
#             y_vals.append(label)  
#     return images, y_vals

# def features_extraction(image, black_background):
#     features = []                                                     # features vector                     

#     height, width, _ = image.shape
#     features.append(height / width)

#     # get gray and binary version of image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary_img = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY)

#     if black_background:
#         binary_img = 1-binary_img

#     pixels_count = height * width
#     white_pixels = cv2.countNonZero(binary_img)                        # number of white pixels
#     black_pixels = pixels_count - white_pixels                         # number of black pixels
#     features.append(black_pixels / white_pixels)

#     # Calculate vertical and horizontal transitions count
#     horizontal_transition_count = 0
#     vertical_transition_count = 0
#     VerticalLastPixel = np.ones(width)
#     lastPixel = 1
#     for row in range(height):
#             for col in range(width):
#                 if(lastPixel != binary_img[row,col]):
#                     horizontal_transition_count += 1
#                 lastPixel = binary_img[row,col]
#                 if(VerticalLastPixel[col] != binary_img[row,col]):
#                     vertical_transition_count += 1
#                 VerticalLastPixel[col] = binary_img[row,col]
#     # append vertical and horizontal transitions count to feature vector
#     features.append(vertical_transition_count)
#     features.append(horizontal_transition_count)

#     quarter_pixels = pixels_count / 4
#     W1 = cv2.countNonZero(binary_img[0:int(height/2), 0:int(width/2)])           # number of white pixels in first quarter of image
#     B1 = quarter_pixels - W1                                                     # number of black pixels in first quarter of image
#     W2 = cv2.countNonZero(binary_img[0:int(height/2), int(width/2):width])           
#     B2 = quarter_pixels - W2     
#     W3 = cv2.countNonZero(binary_img[int(height/2):height, 0:int(width/2)])           
#     B3 = quarter_pixels - W3
#     W4 = white_pixels - (W1 + W2 + W3)
#     B4 = quarter_pixels - W4 

#     # avoid division by zero
#     if W1 == 0:
#         W1 =1 
#     if W2 == 0:
#         W2 =1 
#     if W3 == 0:
#         W3 =1 
#     if W4 == 0:
#         W4 =1 
#     if B2 == 0:
#         B2 =1 
#     if B3 == 0:
#         B3 =1 
#     if B4 == 0:
#         B4 =1 

#     features.extend([B1/W1, B2/W2, B3/W3, B4/W4, B1/B2, B1/B3, B1/B4, B2/B3, B2/B4, B3/B4])

#     # Determine center of mass (of black ink)
#     sx = 0
#     sy = 0
#     for y in range(height):
#         for x in range(width):
#             if np.all(binary_img[y,x] == 0):
#                 sx = sx + x
#                 sy = sy + y
#     if black_pixels == 0:
#         black_pixels =1
#     centerX = int(sx / black_pixels)
#     centerY = int(sy / black_pixels)

#     features.extend([centerX/width, centerY/height])

#     # Calculate horizontal and vertical histogram
#     if not black_background:
#         gray = 255-gray
#     img_row_sum = np.sum(gray, axis=1).tolist()                                     
#     img_col_sum = np.sum(gray, axis=0).tolist()                   
#     img_row_sum[:] = [x / width for x in img_row_sum]                                # normalization
#     img_col_sum[:] = [x / height for x in img_col_sum]

#     img_row_sum_split = np.array_split(img_row_sum, 20)                              # split into 20 chunks
#     img_col_sum_split = np.array_split(img_col_sum, 20)

#     # append the average of the 20 chunks to feature vector (for horizontal and vertical)
#     for chunk in img_col_sum_split:                                                  
#         if chunk.size != 0:
#             features.append(np.average(chunk))
#         else:
#             features.append(0)
#     for chunk in img_row_sum_split:
#         if chunk.size != 0:
#             features.append(np.average(chunk))
#         else:
#             features.append(0)
#     return features

class SVM():

    def __init__(self, featuresNumber):
        super().__init__()

        self.x_vals = [] # Features from input images
        self.y_vals = [] # Labels of input images
        
        self.x_train_vals = [] # features for training dataset
        self.y_train_vals = [] # training dataset labels
        
        self.x_test_vals = []  # features for testing dataset
        self.y_test_vals = []  # testing dataset labels


    def train(self):
        self.x_train_vals, self.x_test_vals, self.y_train_vals, self.y_test_vals = train_test_split(self.x_vals, self.y_vals, test_size=0.30, random_state=42)
        self.x_vals_train = np.array(self.x_train_vals)
        self.x_vals_test = np.array(self.x_test_vals)
        self.y_vals_train = np.array(self.y_train_vals)
        self.y_vals_test = np.array(self.y_test_vals)
        n_classes = len(set(self.y_vals_train))
        self.classifier = learn.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=self.x_vals_train.shape[1])],
                                            n_classes=n_classes)
        self.classifier.fit(self.x_vals_train, self.y_vals_train, steps=5000)
        # self.classifier.save('Model/SVM.sav')

    def test(self):
        y_pred = self.classifier.predict(self.x_vals_test)
        print(metrics.classification_report(y_true=self.y_vals_test, y_pred=list(y_pred)))

    def saveModel(self, fileName):
        # save the model to disk
        # joblib.dump(self.classifier, fileName + '.sav')
        pickle.dump(self.classifier, open(fileName + '.sav', 'w'))

    def loadModel(self,fileName):
        # load the model from disk
        # self.classifier = joblib.load(fileName + '.sav')
        self.classifier = pickle.load(open(fileName + '.sav', 'r'))

    def getResult(self, x):
        y_pred = self.classifier.predict(self.x_vals_test)
        return y_pred
        
# if __name__ == "__main__":    
    # images = read_images_in_folder('/home/ahmed/Desktop/LettersDataset')

    # data = len(images)
    # for i in tqdm(range(data)):
    #     image = images[i]
    #     x_vals.append(features_extraction(image, False))
        
    

