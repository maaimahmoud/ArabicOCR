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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC , LinearSVC

# import joblib
import pickle
chunk_size = 100
class SVMRBF():

    def __init__(self, featuresNumber):
        super().__init__()

        self.x_vals = [] # Features from input images
        self.y_vals = [] # Labels of input images
        
        self.x_train_vals = [] # features for training dataset
        self.y_train_vals = [] # training dataset labels
        
        self.x_test_vals = []  # features for testing dataset
        self.y_test_vals = []  # testing dataset labels
        self.scaler = MaxAbsScaler()

    def train(self):
        self.x_train_vals, self.x_test_vals, self.y_train_vals, self.y_test_vals = train_test_split(self.x_vals, self.y_vals, test_size=0.2, random_state=42)
        
        self.x_train_vals = self.scaler.fit_transform(self.x_train_vals)

        self.x_test_vals = self.scaler.transform(self.x_test_vals)

       
        # get best parameters for SVM classifier
        self.classifier = self.getBestParams()

        # self.classifier = SVC(C = 70, gamma = 0.4, cache_size= 10000, 
        #                         decision_function_shape= 'ovr')
    
        # self.classifier.fit(self.x_train_vals, self.y_train_vals)

    def test(self):

        y_predict=self.clf.predict(self.x_test_vals)
        labels = self.y_train_vals
        labels.extend(self.y_test_vals)

        labels=sorted(list(set(np.array(labels))))
        print("\nConfusion matrix:")
        print("Labels: {0}\n".format(",".join(str(labels))))
        print(confusion_matrix(self.y_test_vals, y_predict, labels=labels))
    
        print("\nClassification report:")
        print(classification_report(self.y_test_vals, y_predict))

    def saveModel(self, fileName):
        # save the model to disk
        # joblib.dump(self.classifier, fileName + '.sav')
        save_data = [self.clf, self.scaler]
        pickle.dump(save_data, open((fileName + '.sav'), 'wb+'))

    def loadModel(self,fileName):
        # load the model from disk
        # self.classifier = joblib.load(fileName + '.sav')
        load_data = pickle.load(open((fileName + '.sav'), 'rb'))
        self.clf = load_data[0]
        self.scaler = load_data[1]
        
    def getResult(self, x):
        x_test = self.scaler.transform(x)
        y_pred = self.clf.predict(x_test)
        return y_pred

    def getBestParams(self):
        # candidate parameters to be evaluated
        param = [
            {
                "kernel": ["rbf"],
                "C": [1, 70, 100],
                "gamma": [0.01, 0.4, 1] 
            }
        ]
        # request one-vs-all strategy
        svm = SVC(cache_size= 10000, decision_function_shape= 'ovr')
    
        # 5-fold cross validation, each parameter set is trained in parallel
        clf = GridSearchCV(svm, param,
                cv=5, n_jobs=-1, verbose=5)
    
        for chunk in range(0,chunk_size, len(self.x_train_vals) ):
            clf.fit(self.x_train_vals[chunk:min(len(self.x_train_vals), chunk+chunk_size)], self.y_train_vals[chunk:min(len(self.x_train_vals), chunk+chunk_size)])
            save_data = [self.clf, self.scaler]
            pickle.dump(save_data, open(('Models/SVMRBF-chunk'+str(chunk) + '.sav'), 'wb+'))
        
        print("\nBest parameters set:")
        print(clf.best_params_)
        return clf
        
# if __name__ == "__main__":    
    # images = read_images_in_folder('/home/ahmed/Desktop/LettersDataset')

    # data = len(images)
    # for i in tqdm(range(data)):
    #     image = images[i]
    #     x_vals.append(features_extraction(image, False))
        
    

