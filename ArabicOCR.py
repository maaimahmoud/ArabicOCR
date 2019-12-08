# To Run Arabic OCR
# python ArabicOCR.py <featureMethod> <classifier>
# featureMethod: StatisticalFeatures - NewGeometricFeatures
# classifier: SVM

TRAINING_DATASET = './Letters-Dataset-Generator/LettersDataset'
#TRAINING_DATASET = './Dataset/scanned'
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
        # only read 6 images
        if image_count == 6:
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
        words += [WordSegmentation(lines[i], i, saveResults=False)]

    #print(len(words), len(words[0]), len(words[1]))
    characters = []
    # Segment words into characters
    for i in range(len(words)):
        currentLine = []
        for j in range(len(words[i])):
            currentLine += [CharacterSegmentation(cv2.cvtColor(np.array(words[i][j], dtype=np.uint8), cv2.COLOR_GRAY2BGR), lineNumber=i, wordNumber =j)]
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
    classifier = classifierClass()
    ###########################

    mode = int(input("1.Train\n2.Test existing Model\n"))

    if mode == 1:
        # set start time
        start_time = timeit.default_timer()

        #trainingImages, classifier.y_vals, filesNames = readImagesInFolder(TRAINING_DATASET)
        trainingImages, classifier.y_vals, __ = readImages(TRAINING_DATASET, trainTest = 0)
        print('-----------------------------')
        
        print('Feature Extraction Phase')
        if TRAINING_DATASET == './Dataset/scanned':
            image_count = 0
            exit_loop = False
            trainingImages = []
            for r, d, f in os.walk(TRAINING_DATASET):
                for file in f:
                    if '.png' in file:
                        # only read first 6 images
                        if image_count == 6 :
                            exit_loop = True
                            break
                        image_count += 1
                        # files.append(os.path.join(r, file))
                        trainingImages += [cv2.imread(TRAINING_DATASET+'/'+file)]
                        # print(TRAINING_DATASET+'/'+file)
                if exit_loop:
                    break

            # print(len(trainingImages))
            for i in tqdm(range(len(trainingImages))):
                image = trainingImages[i]
                # print(image)
                segmented = imagePreprocessing(image) # Get characters of image
                
                # print("Preprocessed")
                textFileName = filesNames[0][:-4].replace('scanned','text')
                textFile = open("text/" + textFileName+'.txt', encoding='utf-8')
                # # textCharacters = list(textFile.read().replace('\n',''))
                # print(get_labels(textFile.read().replace('\n','')))
                classifier.y_vals.append(get_labels(textFile.read().replace('\n','')))
                # [[[, , , characters], , , words] , , , lines]
                for line in segmented:
                    for word in line:
                        for char in word:
                            #print('Currently processing image '+filesNames[0]+' line #', segmented.index(line), ' word #', line.index(word),' char #', word.index(char))
                            currentCharFeature = features.getFeatures(char, False)
                            classifier.x_vals.append(currentCharFeature)
                        # f.write(' ')
                    # f.write('\n')
                # f.close()
                filesNames.pop(0)

        else:
            # Get Features
            for i in tqdm(range(len(trainingImages))):
                image = trainingImages[i]
                classifier.x_vals.append(features.getFeatures(image, False))
        print('-----------------------------')

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