from enum import Enum

# label value => [isolated, end, middle, start]
class Letter(Enum):
    ا = [0,1]
    أ = [2,3]
    آ = [4,5]
    إ = [6,7]
    ب = [8,9,10,11]
    ت = [12,13,14,15]
    ث = [16,17,18,19]
    ج = [20,21,22,23]
    ح = [24,25,26,27]
    خ = [28,29,30,31]
    د = [32,33]
    ذ = [34,35]
    ر = [36,37]
    ز = [38,39]
    س = [40,41,42,43]
    ش = [44,45,46,47]
    ص = [48,49,50,51]
    ض = [52,53,54,55]
    ط = [56,57,58,59]
    ظ = [60,61,62,63]
    ع = [64,65,66,67]
    غ = [68,69,70,71]
    ف = [72,73,74,75]
    ق = [76,77,78,79]
    ك = [80,81,82,83]
    ل = [84,85,86,87]
    م = [88,89,90,91]
    ن = [92,93,94,95]
    ه = [96,97,98,99]
    و = [100,101]
    لا = [102,103]
    لأ = [104, 105]
    ﻵ = [106,107]
    لإ = [108, 109]
    ى = [110,111]
    ي = [112,113,114,115]
    ة = [116,117]
    ء = [118]
    ؤ = [119,120]
    ئ = [121,122,123,124]
    INVALID = 125

terminal_characters = ["ا", "أ", "آ", "إ", "د", "ذ", "ر", "ز", "و", "ؤ", "ة", "ء"]

def get_labels(word):
    labels = []
    invalid_end = False
    i = 0
    if word[-1] not in Letter.__members__:
        word = word[:-1]
        invalid_end = True
    while i < len(word):
        if word[i] in Letter.__members__:
            label = Letter[word[i]].value
            if word[i] == "ل" and i+1 < len(word) and word[i+1] in ["ا", "أ", "آ", "إ"]:   # special case "لا"
                comb = word[i] + word[i+1]
                if i == 0 or word[i-1] in terminal_characters:
                    label = Letter[comb].value[0]
                else:
                    label = Letter[comb].value[1]
                labels.append(label)
                i += 1
            elif i == 0:
                if len(word) == 1 or word[i] in terminal_characters:
                    labels.append(label[0])
                else:
                    labels.append(label[3])
            elif i == len(word)-1:
                if word[i-1] in terminal_characters:
                    labels.append(label[0])
                else:
                    labels.append(label[1])
            else:
                if word[i-1] in terminal_characters:
                    if word[i] in terminal_characters:
                        labels.append(label[0])
                    else:
                        labels.append(label[3])
                else:
                    if word[i] in terminal_characters:
                        labels.append(label[1])
                    else:
                        labels.append(label[2])
        else:
            labels.append(Letter.INVALID.value)
        i += 1
    if invalid_end:
        labels.append(Letter.INVALID.value)
        invalid_end = False
    return labels

if __name__ == "__main__":    
    #document = "نحب الصيف والشتاء لأن الجو جميلا"
    #print(get_labels(document))
    pass