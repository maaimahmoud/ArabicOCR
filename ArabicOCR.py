# To Run Arabic OCR
# python ArabicOCR.py <featureMethod> <classifier>
# featureMethod: StatisticalFeatures - NewGeometricFeatures
# classifier: SVM

TRAINING_DATASET = './Letters-Dataset-Generator/LettersDataset'
# TRAINING_DATASET = './Dataset/scanned'
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
from Classification.TextLabeling import get_labels, getCharFromLabel
# from Preprocessing.PreprocessingTrain import get_dataset

import h5py
hdf5_dir = "PreprocessingOutput/1000-2000/"
def get_dataset(chars_file, labels_file):
    cfile = h5py.File(hdf5_dir + chars_file, "r+")
    imgs = []
    for img in cfile.keys():
        words = []
        for word in cfile[img].keys():
            word_1 = []
            for char in cfile[img][word].keys():
                word_1 += [np.array(cfile[img][word][char])]
            words += [word_1]
        imgs += [words]

    lfile = h5py.File(hdf5_dir + labels_file, "r+")
    labels = []
    for img in lfile.keys():
        label_img = []
        for word in lfile[img].keys():
            label_1 = []
            for label in lfile[img][word].keys():
                label_1 += [np.array(lfile[img][word][label])]
            label_img += [label_1]
        labels += [label_img]

    return imgs, labels


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
        words.extend(WordSegmentation(lines[i], lineNumber = i, saveResults=False))

    characters = []
    # Segment words into characters
    for word in words:
        characters.append(CharacterSegmentation(np.array(word, dtype=np.uint8)))

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
            TotalImages = 0

            for i in tqdm(sorted(glob.glob(TRAINING_DATASET + "*/*.png"),  key=natural_keys)[:2]):
                image = cv2.imread(i)

                textFileName = i[:-4].replace('scanned','text')
                textWords = open(textFileName+'.txt', encoding='utf-8').read().replace('\n',' ').split(' ')

                textWords = [item for item in textWords if item != '']

                segmented = imagePreprocessing(image) # Get characters of image
                # print("segmented", textFileName)
                
                # [[[, , , characters], , , words] , , , lines]
                double_char = "لا"
                segmentedWords = len(segmented)

                TotalImages += 1
                # print("Text: ", len(textWords), "Segmented: ", len(segmented))
                if len(textWords) > segmentedWords:
                    skippedImages += 1
                    continue
                faultyWordSegmented = False
                if len(textWords) < segmentedWords:
                    print("FAULTY IMAGE")
                    faultyWordSegmented = True


                for wordIndex in range(len(segmented)):

                    word = segmented[wordIndex]

                    correspondingTextWord = textWords[0]

                    # get count of occurances of "lam-alf" in word
                    occurances_count = correspondingTextWord.count(double_char)

                    text_length = len(correspondingTextWord)
                    
                    # treat every "lam-alf" as one character
                    text_length -= occurances_count

                    # print("NOW Processing: ", correspondingTextWord, "text_length = ", text_length, " occurances_count = ", occurances_count, ' len(word) = ', len(word) )

                    if len(word) != text_length: # segmented characters != word characters
                        if faultyWordSegmented and wordIndex + 1 < len(segmented) and occurances_count > 0 and  text_length == len(word) + len(segmented[wordIndex+1]): 
                            # There is لا that is causing a problem
                            correspondingTextWord = correspondingTextWord[:len(word)+1]
                            print("correspondingTextWord = ", correspondingTextWord)
                            textWords[0] = textWords[0][len(word)+1:]
                            processedWords += 1
                        else:
                            ignoredWords += 1
                            del textWords[0]
                            continue
                    else:
                        processedWords += 1
                        del textWords[0]
                        
                    classifier.y_vals.extend(get_labels(correspondingTextWord))
                    for char in word:
                        processedCharacters += 1
                        #print('Currently processing image '+filesNames[0]+' line #', segmented.index(line), ' word #', line.index(word),' char #', word.index(char))
                        currentCharFeature = features.getFeatures(char, showResults = False, black_background=True)
                        classifier.x_vals.append(currentCharFeature) #cv2.resize(char, (100,60))
            
            print("processedCharacters = ", processedCharacters, "Characters from text = ", len(classifier.y_vals))
            print("ignoredWords = ", ignoredWords, " processedWords = ", processedWords)
            print("skipped Images = ", skippedImages, " (out of ", TotalImages,")")
            print('-----------------------------')
        else:
            # trainingImages, classifier.y_vals, filesNames = readImages(TRAINING_DATASET, 0)
            print("Loading dataset")
            trainingImages, labels = get_dataset('chars_101.h5', 'labels_101.h5')
            print("Finished Loading dataset")
            # Get Features
            # for i in tqdm(range(len(trainingImages))):
            for i in range(len(trainingImages)):
                print(i)
                for j in range(len(trainingImages[i])):
                    for k in range(len(trainingImages[i][j])):
                        if (len(trainingImages[i][j][k]) == len(labels[i][j][k])):
                            image = np.array(trainingImages[i][j][k])
                            classifier.x_vals.append(features.getFeatures(image, False))
                            classifier.y_vals.append(labels[i][j][k])
        
        # Train classifer
        print('Training Phase')
        print('-----------------------------')
        classifier.train()
        
        # for i in tqdm(sorted(glob.glob(TESTING_DATASET + "*/*.png"))):
        #     image = cv2.imread(i)
        #     textFileName = i[:-4]+'.txt'#.replace('scanned','text')
        #     segmented = imagePreprocessing(image) # Get characters of image
        #     print(len(segmented))
        #     # [[[, , , characters], , , words] , , , lines]
        #     f = open(textFileName,'w') 
        #     for word in segmented:
        #         for char in word:
        #             currentCharFeature = features.getFeatures(char, False)
        #             classificationResult = classifier.getResult([currentCharFeature])
        #             # char = 'أ'
        #             char = getCharFromLabel(classificationResult)
        #             f.write(char)
        #         f.write(' ')
        #         # f.write('\n')
        #     f.close()

        # Test Model
        print('Testing Phase')
        print('-----------------------------')
        classifier.test()

        # Calculate and print total runtime
        print('Runtime: ', (timeit.default_timer() - start_time)/60) 

        # Save Model
        print('Model Saved as ' +'Models/'+ args.classifier+'-'+args.features+'.sav')
        classifier.saveModel('Models/'+args.classifier+'-'+args.features)

    else:
        # modelFileName = input("Model filename:")
        print('Loading Model')
        print('-----------------------------')
        
        classifier.loadModel('Models/'+args.classifier+'-'+args.features)

        print('Load Dataset Phase')
        print('-----------------------------')
        # trainingImages, __ , filesNames = readImages(TESTING_DATASET, trainTest = 1)
        # print(filesNames)

        print('Processing')
        print('-----------------------------')

        for i in tqdm(sorted(glob.glob(TESTING_DATASET + "*/*.png"))):
            image = cv2.imread(i)
            textFileName = i[:-4]+'.txt'#.replace('scanned','text')
            segmented = imagePreprocessing(image) # Get characters of image
            print(len(segmented))
            # [[[, , , characters], , , words] , , , lines]
            f = open(textFileName,'wb+') 
            for word in segmented:
                for char in word:
                    currentCharFeature = features.getFeatures(char, False)
                    classificationResult = classifier.getResult([currentCharFeature])
                    # char = 'أ'
                    char = getCharFromLabel(classificationResult)
                    f.write(char.encode('utf8'))
                f.write(' '.encode('utf8'))
                # f.write('\n')
            f.close()
            # filesNames.pop(0)