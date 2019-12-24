# To Run Arabic OCR
# python ArabicOCR.py <featureMethod> <classifier>
# featureMethod: StatisticalFeatures - NewGeometricFeatures
# classifier: SVM

# TRAINING_DATASET = './Letters-Dataset-Generator/LettersDataset'
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
from Classification.TextLabeling import get_labels, getCharFromLabel
# from Preprocessing.PreprocessingTrain import get_dataset

import h5py
from multiprocessing import Process
# hdf5_dir = "PreprocessingOutput/1000-2000/"
# def get_dataset(chars_file,labels_file,count=-1):
#     cfile= h5py.File(hdf5_dir +chars_file, "r+")
#     imgs=[]
#     i=0
#     for img in cfile.keys(): 
#         if count!= -1 and i>=count: 
#             break
#         i+=1
#         words=[]
#         word_k= len(cfile[img].keys())
#         for word in range(word_k):
#             word_1=[]
#             for char in cfile[img][str(word)].keys():
#                 word_1+=[np.array(cfile[img][str(word)][char])]
#             words+=[word_1]
#         imgs+=[words]
     
#     lfile= h5py.File(hdf5_dir +labels_file, "r+")
#     labels=[]
#     i=0
#     for img in lfile.keys():
#         if count!= -1 and i>=count: 
#             break
#         i+=1
#         label_img=[]
#         for word in lfile[img].keys():
#             label_1=[]
#             for label in lfile[img][word].keys():
#                 label_1+=[np.array(lfile[img][word][label])]
#             label_img+=[label_1]
#         labels+=[label_img]  


#     return imgs,labels


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
    # print("lines= ",len(lines)," words= ",len(words)," chars= ",len(characters))
    return characters # [[[, , , characters], , , words] , , , lines] 

def loop(i):
    pictureFeatures = []
    pictureLabels = []
    # actualCharacters = []

    image = cv2.imread(i)

    textFileName = i[:-4].replace('scanned', 'text')
    textWords = open(textFileName + '.txt', encoding='utf-8').read().replace('\n', ' ').split(' ')

    textWords = [item for item in textWords if item != '']

    segmented = imagePreprocessing(image)  # Get characters of image
    # print("segmented", textFileName)

    # [[[, , , characters], , , words] , , , lines]
    double_char = "لا"
    segmentedWords = len(segmented)

    # TotalImages += 1
    # print("Text: ", len(textWords), "Segmented: ", len(segmented))
    if len(textWords) != segmentedWords:
        # skippedImages += 1
        # print(i[i.rfind('\\')+1:-4],"is skipped")
        return
    # faultyWordSegmented = False
    # if len(textWords) < segmentedWords:
    #     print("FAULTY IMAGE")
    #     faultyWordSegmented = True

    for wordIndex in range(len(segmented)):

        word = segmented[wordIndex]

        correspondingTextWord = textWords[0]

        # get count of occurances of "lam-alf" in word
        occurances_count = correspondingTextWord.count(double_char)

        text_length = len(correspondingTextWord)

        # treat every "lam-alf" as one character
        text_length -= occurances_count

        # print("NOW Processing: ", correspondingTextWord, "text_length = ", text_length, " occurances_count = ", occurances_count, ' len(word) = ', len(word) )

        if len(word) != text_length:  # segmented characters != word characters
            # if faultyWordSegmented and wordIndex + 1 < len(segmented) and occurances_count > 0 and  text_length == len(word) + len(segmented[wordIndex+1]):
            #     # There is لا that is causing a problem
            #     correspondingTextWord = correspondingTextWord[:len(word)+1]
            #     print("correspondingTextWord = ", correspondingTextWord)
            #     textWords[0] = textWords[0][len(word)+1:]
            #     processedWords += 1
            # else:
            # ignoredWords += 1
            del textWords[0]
            continue
        # else:
        #     processedWords += 1
        #     del textWords[0]

        pictureLabels.extend(get_labels(correspondingTextWord))
        # actualCharacters += [correspondingTextWord]
        for char in word:
            # processedCharacters += 1
            # print('Currently processing image '+filesNames[0]+' line #', segmented.index(line), ' word #', line.index(word),' char #', word.index(char))
            currentCharFeature = features.getFeatures(char, showResults=False, black_background=True)
            pictureFeatures.append(currentCharFeature)  # cv2.resize(char, (100,60))
        del textWords[0]
    # f = open('textFiles/'+i[i.rfind('\\')+1:-4]+'-words.txt','wb+')
    # for myWord in actualCharacters:
    #     f.write(myWord.encode('utf8')+'\n'.encode('utf8'))
    # f.close()

    f = open('textFiles/'+i[i.rfind('\\')+1:-4]+'.txt','w+')
    for k in range(len(pictureFeatures)):
        f.write(str(pictureLabels[k])+' ')
        for current_feature in pictureFeatures[k]:
            f.write("%s " % current_feature)

        f.write('\n')
    f.close()
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


if __name__ == "__main__":

    # Import classifier Type
    classifierModule = import_module('Classification.' + args.classifier)  # Dynamically load the classifier module
    classifierClass = getattr(classifierModule, args.classifier)
    classifier = classifierClass(features.featuresNumber)
    ###########################

    mode = int(input("1.Segment\n2.Train\n3.Test existing Model\n"))

    if mode == 1:
        # set start time
        start_time = timeit.default_timer()

        #trainingImages, classifier.y_vals, __ = readImages(TRAINING_DATASET, trainTest = 0)
        
        # if TRAINING_DATASET == './Dataset/scanned':
        # os.listdir("Dataset/scanned/")
        print("Reading  dataset to segment")

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

        processes = []
        start_time = timeit.default_timer()
        dataset = sorted(glob.glob(TRAINING_DATASET + "*/*.png"),  key=natural_keys)[4400:4600] # 3000 DONE

        for i in list(dataset):
            p = Process(target=loop, args=(i,))
            p.start()
            processes += [p]

        for p in processes:
            p.join()

        print("running time = ", timeit.default_timer() - start_time)
        print('Finished All#########################################')

    elif mode==2:
        featuresList=list(glob.glob("textFiles" + "/*.txt"))[:1000]
        for filepath in tqdm(featuresList):
            with open(filepath) as fp:
                for line in fp:
                    charData=line.replace('\n','').split(' ')
                    del charData[-1]
                    label=int(charData[0])
                    charData= [float(i) for i in charData[1:len(charData)]]
                    # print(len(charData),label,charData)
                    classifier.x_vals.append(charData)
                    classifier.y_vals.append(label)
        print("Done reading segmented files")
        print('-----------------------------')
        print('Training Phase')
        print('-----------------------------')
        classifier.train()

        print('Testing Phase')
        print('-----------------------------')
        classifier.test()

        # Calculate and print total runtime
        print('Runtime: ', (timeit.default_timer() - start_time)/60) 

        # Save Model
        print('Model Saved as ' +'Models/'+ args.classifier+'-'+args.features+ '-50' + '.sav')
        classifier.saveModel('Models/'+args.classifier+'-'+args.features+ '-50')

    else:
        # modelFileName = input("Model filename:")
        print('Loading Model')
        print('-----------------------------')
        
        classifier.loadModel('Models/'+args.classifier+'-'+args.features + '-50')

        print('Load Dataset Phase')
        print('-----------------------------')
        # trainingImages, __ , filesNames = readImages(TESTING_DATASET, trainTest = 1)
        # print(filesNames)

        print('Processing')
        print('-----------------------------')

         # create output directory
        directory = "./output/text/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        runtime_file = open("./output/running_time.txt",'w+')
        # read test images and print corresponding text

        for i in tqdm(sorted(glob.glob(TESTING_DATASET + "*/*.png"))):
            textFileName = os.path.basename(i)[:-4]+'.txt'#.replace('scanned','text')
            f = open(directory + textFileName,'wb+') 
            image = cv2.imread(i)
            start_time = timeit.default_timer()     # start timer
            segmented = imagePreprocessing(image) # Get characters of image
            # print(len(segmented))
            # [[[, , , characters], , , words] , , , lines]
            for word in segmented:
                for char in word:
                    currentCharFeature = features.getFeatures(char, False)
                    classificationResult = classifier.getResult([currentCharFeature])
                    # char = 'أ'
                    char = getCharFromLabel(classificationResult)
                    f.write(char.encode('utf8'))
                f.write(' '.encode('utf8'))
                # f.write('\n')
            runtime_file.write(str(timeit.default_timer() - start_time) + '\n')  # write running time to file
            f.close()
            # filesNames.pop(0)
        runtime_file.close() 
