import h5py
import cv2
import numpy as np
import os
import re
import glob
from tqdm import tqdm
# from pathlib import Path

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

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



hdf5_dir = "../PreprocessingOutput/"
disk_dir = "../PreprocessingOutput/LineSegmentation/"
lines_file="lines.h5"
words_file="words.h5"
chars_file="chars.h5"
# disk_dir.mkdir(parents=True, exist_ok=True)
# hdf5_dir.mkdir(parents=True, exist_ok=True)

linesList=list(sorted(glob.glob(disk_dir+"*.png"),  key=natural_keys)[:]) 
maxPortion=len(linesList)
images,labels=[],[]
for i in linesList[0:2]:
    images+= [cv2.imread(i)]
    # labels+=[i[ i.rfind('/') + 1 : -4]]

images=np.array(images)
# labels=np.array(labels)
# Create a new HDF5 file
file = h5py.File(hdf5_dir +lines_file, "w")
# print(images.shape, labels.shape, images[0].shape, labels[0].shape)
store_many_hdf5(images,str(1),file)
store_many_hdf5(images,str(2),file)

file.close()
images2,labels2=[],[]
images0,labels0 =read_many_hdf5(lines_file,str(1))
images1,labels1 =read_many_hdf5(lines_file,str(2))
print(labels0,labels1)
images2=images0+images1
labels2=labels0+labels1
print("looping over imgs")
for i in range(len(images2)):
    print("../PreprocessingOutput/"+labels2[i]+".jpg",images2[i].shape)
    cv2.imwrite("../PreprocessingOutput/"+labels2[i]+".jpg",images2[i])



