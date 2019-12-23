# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('../Classification')
import h5py
import cv2
import numpy as np
from Lines import LineSegmentation
from Words import WordSegmentation
from Characters import CharacterSegmentation
from TextLabeling import get_labels
import re
import glob
from tqdm import tqdm

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#File paths to work with
hdf5_dir = "../PreprocessingOutput/"
TRAINING_DATASET = '../Dataset/scanned'
TRAINING_TEXT='../Dataset/text/'


datasetList=list(sorted(glob.glob(TRAINING_DATASET + "*/*.png"),  key=natural_keys)[5000:7000])
maxPortion=len(datasetList)


def store_many_hdf5(images,imgName,file):
    
    img = file.create_group(str(imgName))
    for i in range(len(images)):
        img.create_dataset(
        str(i), np.shape(images[i]), h5py.h5t.STD_U8BE, data=images[i]
        )


def read_many_hdf5(file_name,img):
 
    images = []
    labels=[]

    # Open the HDF5 file
    file = h5py.File(hdf5_dir + file_name, "r+")

    for data in file[img].keys():
        images += [np.array(file[img][data])]
        labels+=[str(img)+str(data)]
    # labels = np.array(file["/meta"]).astype("uint8")

    return images,labels



def get_dataset(chars_file,labels_file):
    cfile= h5py.File(hdf5_dir +chars_file, "r+")
    imgs=[]
    for img in cfile.keys():  
        words=[]
        for word in cfile[img].keys():
            word_1=[]
            for char in cfile[img][word].keys():
                word_1+=[np.array(cfile[img][word][char])]
            words+=[word_1]
        imgs+=[words]
     
    lfile= h5py.File(hdf5_dir +labels_file, "r+")
    labels=[]
    for img in lfile.keys():
        label_img=[]
        for word in lfile[img].keys():
            label_1=[]
            for label in lfile[img][word].keys():
                label_1+=[np.array(lfile[img][word][label])]
            label_img+=[label_1]
        labels+=[label_img]  


    return imgs,labels




def preprocess(mode,lines_file,words_file="",chars_file="",labels_file="",datasetPortion=maxPortion):

    #number of images to try over

    originalNumberOfWords = 0
    lostNumberOfWords = 0
    wrongSegmented = 0
    correctrlySegmented = 0

    j  = 1
    if mode=="lines": #lost 8 pages 21/12/2019
        if os.path.exists(hdf5_dir +lines_file):
            print("file already exists, change name or delete")
            return
        file = h5py.File(hdf5_dir +lines_file, "w")
        print("starting line seg")
        for i in datasetList[0:datasetPortion]:
            print(i[ i.rfind('\\') + 1 : -4])
            img = cv2.imread(i)
            lines=[]
            lines = LineSegmentation(img, imgName = i[ i.rfind('\\') + 1 : -4] ,saveResults=False)
            if len(lines)==0:
                wrongSegmented+=1
            store_many_hdf5(lines,i[ i.rfind('\\') + 1 : -4],file)
            j+=1
        file.close()
        print("Total Wrongly segmented pages (LineSeg) = ",wrongSegmented)
   
    elif mode=="word":
        if os.path.exists(hdf5_dir +words_file):
            print("file already exists, chaneg name or delete")
            return
        wfile= h5py.File(hdf5_dir +words_file, "w")
        lfile = h5py.File(hdf5_dir + lines_file, "r+")
        print("starting word seg")
        for i in datasetList[0:datasetPortion]:
            img=i[ i.rfind('\\') + 1 : -4]
            print(img)
            lines,_= read_many_hdf5(lines_file,img)
            textWords = open(TRAINING_TEXT+img+'.txt', encoding='utf-8').read().replace('\n',' ').split(' ')
            original = len(textWords)
            calculated= 0
            words = []
            for k in range(len(lines)):
                words += WordSegmentation(lines[k], imgName =  img , lineNumber = k + 1 , saveResults=False)
            print(img,len(words))
            calculated = len(words)            
            if original != calculated:
                wrongSegmented += 1
            else:
                correctrlySegmented += 1
            store_many_hdf5(words,img,wfile)

        print("Wrongly Segmented Pages (WordSeg)= ",wrongSegmented)
        print("Correctly Segmented pages (WordSeg)= ",correctrlySegmented)
        wfile.close()
    elif mode=="char":
        if os.path.exists(hdf5_dir +chars_file) or os.path.exists(hdf5_dir +labels_file):
            print("file already exists, chaneg name or delete")
            return
        wfile= h5py.File(hdf5_dir +words_file, "r+")
        cfile = h5py.File(hdf5_dir + chars_file, "w")
        labelfile= h5py.File(hdf5_dir + labels_file, "w")
        hopefulMoments=0
        dreadful=0

        for i in datasetList[0:datasetPortion]:
            img=i[ i.rfind('\\') + 1 : -4]
            char_count=0
            text_char_count=0
            print(img)
            words,_= read_many_hdf5(words_file,img)
            textWords = open(TRAINING_TEXT+img+'.txt', encoding='utf-8').read().replace('\n',' ').split(' ')
            text_char_count=sum([len(i) for i in textWords])
            characters=[]
            for j in range(len(words)):
                charList=CharacterSegmentation(np.array(words[j], dtype=np.uint8), imgName = img, lineNumber=1, wordNumber = j + 1 , saveResults = False)
                characters += [charList]
                char_count+=len(charList)
            img_grp=cfile.create_group(img) 
            img_grp_l=labelfile.create_group(img)
            if(len(words)==len(textWords)):
                #correctly segmented procedd to labeleach char with word labels
                for i in range(len(words)):
                    store_many_hdf5(characters[i],str(i),img_grp)
                    w_labels=get_labels(textWords[i])
                    if(len(w_labels)==len(characters[i])): 
                        correctrlySegmented += 1
                    else:
                        wrongSegmented += 1
                    store_many_hdf5(w_labels,str(i),img_grp_l)
            else:
                if text_char_count==char_count:
                    #there is hope!
                    hopefulMoments+=1
                    all_chars=[]
                    for char_arrays in characters:
                        all_chars+=char_arrays
                    prev=0
                    for i in range(len(textWords)):
                        store_many_hdf5(all_chars[prev:len(textWords[i])],str(i),img_grp)
                        w_labels=get_labels(textWords[i])
                        store_many_hdf5(w_labels,str(i),img_grp_l)
                else:
                    dreadful+=1

        print("Wrongly Segmented words (charSeg)= ",wrongSegmented)
        print("Correctly Segmented words (charSeg)= ",correctrlySegmented)
        print("pages saved by charseg= ", hopefulMoments)
        print("Pages that couldn't be saved",dreadful)
        cfile.close()
        labelfile.close()
    

if __name__ == "__main__":

    # specify dataset portion to segment over max is default
    portion=2000
    #specify mode to work with, lines/word/char
    # to work with words, lines must already exist, to work with char, words must exist
    mode="lines"
    #specify file names, they will be saved in dir specified above
    lines_file="lines.h5"
    word_file="word_10.h5"
    char_file="chars_101.h5"    
    labels_file="labels_101.h5" # needs to only be supplied when mode = char

 

    preprocess(mode,lines_file,word_file,char_file,labels_file,portion)

    ##FUNCTION TO CALL TO RETRIEVE SEGMENTED DATA IN ARRAYS
    #get_dataset(char_file,labels_file)

    