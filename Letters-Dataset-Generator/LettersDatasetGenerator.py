import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from TextSupport import *
import cv2

FontsDir = 'Fonts'
assert os.path.isdir(FontsDir) , 'Please put arabic fonts in the directory /Fonts'

FontsFiles = []
for (_, _, FileNames) in os.walk(FontsDir):
    FontsFiles.extend(FileNames)
    break

DatasetDir = 'LettersDataset'

if os.path.isdir(DatasetDir) is False:
    os.mkdir(DatasetDir)

# 064A
LettersFile = open('Letters', 'r')
LettersFileText = LettersFile.read()
LettersFile.close()
Letters = [Letter for Letter in LettersFileText.split('\n') if Letter != '']
for Index, Letter in enumerate(Letters):
    Letters[Index] = [Input for Input in Letter.split(' ') if Input.replace(' ', '') != '']

id = 0
letter_start_id = 0
count = 0
supported_count = 0

for Letter in Letters:
    id = letter_start_id
    integer = int(Letter[0], 16)

    if len(Letter[1:]) == 0:
        continue

    string = str(integer)
    character = chr(integer)

    if os.path.isdir(DatasetDir + '/' + string) is False:
        os.mkdir(DatasetDir + '/' + string)

    for FontFile in FontsFiles:
        for Index, (Prefix, Text) in enumerate([('Isolated', character), 
            ('End', 'ـ' + character),
            ('Middle', 'ـ' + character + 'ـ'),
            ('Beginning', character + 'ـ')]):
            if len(Letter[1:]) <= Index:
                break

            Text = arabic_reshaper.reshape(Text)
            Text = get_display(Text, base_dir='R')
        
            count += 1
            if isTextSupported(FontsDir + '/' + FontFile, Text) is False:
                continue
            W, H = 50, 50                   # image dimensions
            for size in range(12, 32):      # font size range

                Font = ImageFont.truetype(FontsDir + '/' + FontFile, size)
                Img = Image.new("RGB", (W, H),(255,255,255))
                Draw = ImageDraw.Draw(Img)
                w, h = Draw.textsize(Text, Font)

                Draw.text(((W - w) / 2, (H - h) / 2), Text, (0,0,0), font = Font)
                Draw = ImageDraw.Draw(Img)
                    
                Img = np.array(Img)
                Img = Img[:, :, ::-1].copy()
                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                
                cv2.imwrite(DatasetDir + '/' + string + '/' + FontFile[:FontFile.rfind('.')] + Prefix + "_label_" + str(id) + "_size_" + str(size) + ".png", Img)
            id += 1
        supported_count = count
        count = 0
        id = letter_start_id
    letter_start_id += supported_count