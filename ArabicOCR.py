# To Run Arabic OCR
# python ArabicOCR.py <featureMethod> <classifier>
# featureMethod: StatisticalFeatures - NewGeometricFeatures
# classifier: SVM

#TRAINING_DATASET = './Letters-Dataset-Generator/LettersDataset'
TRAINING_DATASET = './Dataset/scanned'
TESTING_DATASET = './Dataset/Testing'


from importlib import import_module
import argparse
from tqdm import tqdm
import timeit
import glob
import cv2
import numpy as np
import os
from Preprocessing.Lines import LineSegmentation
from Preprocessing.Words import WordSegmentation
from Preprocessing.Characters import CharacterSegmentation
from Classification.TextLabeling import get_labels

import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# checking if string contains list element 
def contains(string, str_list):
    res = [ele for ele in str_list if(ele in string)] 
    return bool(res)

def readImages(folder, trainTest = 0):
    images = []
    folders = [f for f in glob.glob(folder + "**/*", recursive=True)]
    y_vals = []
    filesNames = []
    for folder in folders:
        img_files = [img_file for img_file in glob.glob(folder + "**/*.png", recursive=True)]
        for i in range(len(img_files)):
            filesNames += [img_files[i]]
            img = cv2.imread(img_files[i])    
            images.append(img)
            if trainTest == 0:
                label = int(img_files[i][img_files[i].index("label_") + len("label_"):img_files[i].index("_size")])
                y_vals.append(label)  
    return images, y_vals, filesNames

def readImagesInFolder(folder):
    images = []
    folders = [f for f in glob.glob(folder + "**/*", recursive=True)]
    y_vals = []
    filesNames = []
    image_count = 0
    for img in folders:
        # only read first image
        if image_count == 4:
            return images, y_vals, filesNames
        filesNames += [img[img.index("scanned/") + len("scanned/"):]]
        image_count += 1
    return images, y_vals, filesNames

def imagePreprocessing(img):
    # Segment paragraph into lines    
    lines = LineSegmentation(img, saveResults=False)

    # Segment lines into words
    words = []
    for i in range(len(lines)):
        words += [WordSegmentation(lines[i], lineNumber = i, saveResults=False)]

    #print(len(words), len(words[0]), len(words[1]))
    characters = []
    # Segment words into characters
    for i in range(len(words)):
        currentLine = []
        for j in range(len(words[i])):
            currentLine += [CharacterSegmentation(np.array(words[i][j], dtype=np.uint8), lineNumber=i, wordNumber =j)]
        characters += [currentLine]
    return characters # [[[, , , characters], , , words] , , , lines] 



if __name__ == "__main__":
    # Read arguments in order
    parser = argparse.ArgumentParser("Train Module")
    parser.add_argument("features")
    parser.add_argument("classifier")
    args = parser.parse_args() # Parse the arguments written by the user in the commandline

    # Import Modules
    #################

    # Import Features Type
    featuresModule = import_module('FeatureExtraction.' + args.features) # Dynamically load the features module
    featuresClass = getattr(featuresModule, args.features)
    features = featuresClass()

    # Import classifier Type
    classifierModule = import_module('Classification.' + args.classifier) # Dynamically load the classifier module
    classifierClass = getattr(classifierModule, args.classifier)
    classifier = classifierClass(features.featuresNumber)
    ###########################

    mode = int(input("1.Train\n2.Test existing Model\n"))

    if mode == 1:
        # set start time
        start_time = timeit.default_timer()

        #trainingImages, classifier.y_vals, __ = readImages(TRAINING_DATASET, trainTest = 0)
        
        if TRAINING_DATASET == './Dataset/scanned':
            # os.listdir("Dataset/scanned/")
            print("Reading Dataset")

            trainingImages = []
            imagesNames = []

            # for i in tqdm(sorted(glob.glob(TRAINING_DATASET + "*/*.png"),  key=natural_keys)):
            #     trainingImages += [cv2.imread(i)]
            #     imagesNames += [i[:-4]]

            print('-----------------------------')
            print("Preprocessing and feature Extraction Phase")

            processedCharacters = 0
            ignoredWords = 0
            processedWords = 0
            segmented = None

            skippedImages = 0

            for i in tqdm(sorted(glob.glob(TRAINING_DATASET + "*/*.png"),  key=natural_keys)[0:400]):
                image = cv2.imread(i)

                textFileName = i[:-4].replace('scanned','text')
                textWords = open(textFileName+'.txt', encoding='utf-8').read().replace('\n',' ').split(' ')

                textWords = [item for item in textWords if item != '']

                segmented = imagePreprocessing(image) # Get characters of image
                
                # [[[, , , characters], , , words] , , , lines]
                double_char = "لا"
                segmentedWords = 0
                for line in segmented:
                    segmentedWords += len(line)

                if len(textWords) < segmentedWords:
                    skippedImages += 1
                    continue
                for line in segmented:
                    for word in line:
                        text_length = len(textWords[0])
                        # get count of occurances of "lam-alf" in word
                        occurances_count = textWords[0].count(double_char)
                        # treat every "lam-alf" as one character
                        text_length -= occurances_count
                        if len(word) == text_length: # segmented characters != word characters
                            classifier.y_vals.extend(get_labels(textWords[0]))
                            for char in word:
                                processedCharacters += 1
                                #print('Currently processing image '+filesNames[0]+' line #', segmented.index(line), ' word #', line.index(word),' char #', word.index(char))
                                currentCharFeature = features.getFeatures(char, showResults = False, black_background=True)
                                classifier.x_vals.append(currentCharFeature) #cv2.resize(char, (100,60))
                            processedWords += 1
                        else:
                            ignoredWords += 1
                        textWords.pop(0)
            print("processedCharacters = ", processedCharacters, "Characters from text = ", len(classifier.y_vals))
            print("ignoredWords = ", ignoredWords, " processedWords = ", processedWords)
            print("skipped Images = ", skippedImages, " (out of ", len(trainingImages),")")
            print('-----------------------------')
        else:
            trainingImages, classifier.y_vals, filesNames = readImages(TRAINING_DATASET, 0)
            # Get Features
            for i in tqdm(range(len(trainingImages))):
                image = trainingImages[i]
                classifier.x_vals.append(features.getFeatures(image, False))
        
        # Train classifer
        print('Training Phase')
        print('-----------------------------')
        classifier.train()
        
        # Test Model
        print('Testing Phase')
        print('-----------------------------')
        classifier.test()

        # Calculate and print total runtime
        print('Runtime: ', (timeit.default_timer() - start_time)/60) 

        # Save Model
        # print('Model Saved as '+'Models/'+args.classifier+'-'+args.features+'.sav')
        # classifier.saveModel('Models/'+args.classifier+'-'+args.features)

    else:
        # modelFileName = input("Model filename:")
        print('Loading Model')
        print('-----------------------------')
        
        classifier.loadModel('/Models/'+args.classifier+'-'+args.features)

        print('Load Dataset Phase')
        print('-----------------------------')
        trainingImages, __ , filesNames = readImages(TESTING_DATASET+'/scanned', trainTest = 1)

        print('Processing')
        print('-----------------------------')

        for i in tqdm(range(len(trainingImages))):
            image = trainingImages[i]
            segmented = imagePreprocessing(image) # Get characters of image
            # [[[, , , characters], , , words] , , , lines]
            f = open(filesNames[0],'w') 
            for line in segmented:
                for word in line:
                    for char in word:
                        currentCharFeature = features.getFeatures(char, False)
                        classificationResult = classifier.getResult(currentCharFeature)
                        char = 'أ'
                        # char = getCharFromLabel(classificationResult)
                        f.write(char)
                    f.write(' ')
                f.write('\n')
            f.close()
            filesNames.pop(0)