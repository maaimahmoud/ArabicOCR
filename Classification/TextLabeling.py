from enum import Enum

# label value => [isolated, end, middle, start]
class Letter(Enum):
    ا = [0,1]
    ب = [2,3,4,5]
    ت = [6,7,8,9]
    ث = [10,11,12,13]
    ج = [14,15,16,17]
    ح = [18,19,20,21]
    خ = [22,23,24,25]
    د = [26,27]
    ذ = [28,29]
    ر = [30,31]
    ز = [32,33]
    س = [34,35,36,37]
    ش = [38,39,40,41]
    ص = [42,43,44,45]
    ض = [46,47,48,49]
    ط = [50,51,52,53]
    ظ = [54,55,56,57]
    ع = [58,59,60,61]
    غ = [62,63,64,65]
    ف = [66,67,68,69]
    ق = [70,71,72,73]
    ك = [74,75,76,77]
    ل = [78,79,80,81]
    م = [82,83,84,85]
    ن = [86,87,88,89]
    ه = [90,91,92,93]
    و = [94,95]
    لا = [96,97]
    ي = [98,99,100,101]

terminal_characters = ["ا", "د", "ذ", "ر", "ز", "و"]

def get_labels(word):
    labels = []
    i = 0
    while i < len(word):
        if word[i] in Letter.__members__:
            label = Letter[word[i]].value
            if word[i] == "ل" and i+1 < len(word) and word[i+1] == "ا":   # special case "لا"
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
            print("unexpected text symbol!")
        i += 1
    return labels

if __name__ == "__main__":    
    #document = "الاسلاميون"
    #print(get_labels(document))
    pass