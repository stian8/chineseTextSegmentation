# File for cleaning and putting data into correct format
import re
import sys
import jieba.posseg as pseg
import csv
import json

#INPUT = "truth_result.txt"
#INTERMEDIATE = "truth_clean.txt"
#OUTPUT  = "training.csv"
PUNC = set(['，',',','.','!',':',';','“', '"', '，', '。',
            '、', '！', '；', '？','……', '?', '：', '”'])
CON = set(['的','地', '得']) # what to do about le, bu
COLUMNS = ["Label", "No", "B2", "B1", "F1", "F2", "POSB2", "POSB1",
           "POSF1", "POSF2"] # No being the position in sentence
def main(inp, imed, out, dict_file):
    clean_text(inp, imed)
    create_csv(imed, out)
    create_word_dictionary(inp, dict_file)

def create_word_dictionary(inp, dict_file):
    """Creates a word dictionary from the original data file. Used for encoding
    in the trainer later.
    Args:
        inp: the filename of the original, tagged input file
        dict_file: the filename that the dictionary is to be written to.
    """
    dictionary = {'，': 0, '。': 1, '、': 2, '！': 3, '？': 4, '；': 5, '：': 6}
    index = 7
    with open(inp, mode='rt', encoding="utf8") as f:
        data=f.read().replace('\n', '')
        data = re.sub('[“”、？！/；：／，。\(\)]', '', data)
        for char in data:
            if char not in dictionary:
                dictionary[char] = index
                index += 1
    with open(dict_file, mode='w', encoding='utf-8-sig') as file:
        file.write(json.dumps(dictionary, ensure_ascii=False))
    return

def clean_text(inp, imed):
    write_file = open(imed, mode="w", encoding='utf-8-sig')
    result = ""
    with open(inp, mode='rt', encoding="utf8") as f:
        data=f.read().replace('\n', '')
        data = re.sub(' +', '', data)
        data = re.sub('[”“‘’]', '', data)
        previous = None
        final = ['1'] # based on our definition
        for index, char in enumerate(data): 
            if char in CON:
                try:
                    br = final.pop()
                    final.append('0')
                    final.append(char)
                    final.append(br)
                except IndexError:
                    final.append(char)
            elif char == "/" or char == "／":
                final[-1] = '1'
            elif char == "不" and (data[index - 1] != data[index + 1]) and(data[index + 1] != "/" or data[index + 1] != "／"):
                final.append(char)
                final.append('1')
            elif char == "没" and (data[index + 1] != "/" or data[index + 1] != "／"):
                final.append(char)
                final.append('1')
            elif char == "很" and (data[index + 1] != "/" or data[index + 1] != "／"):
                final.append(char)
                final.append('1')
            elif char == "了" and (data[index + 1] != "/" or data[index + 1] != "／"):
                final[-1] = '1'
                final.append(char)
                final.append('1')
            else:
                final.append(char)
                final.append('0')
        write_file.write(''.join(final))
    write_file.close()

def create_csv(imed, out):
    """ Creates/writes csv (utf-8-sig) with label and features generated from
    the data. The first row of the csv contains names of the features/labels.
    The columns are as follows:["Label", "No", "B3", "B2", "B1", "F1", "F2",
    "F3", "POSB3", "POSB2","POSB1","POSF1", "POSF2", "POSF3"]
    Args:
        imed: filename of intermediate file, in which the characters are
            split by either a 0 or 1, 1 marking the existence of a split,
            0 otherwise.
        out: filename of output csv file, described above.
    Returns:
        None
    """
    write_file = open(out, mode="w", encoding='utf-8-sig')
    w = csv.writer(write_file)
    w.writerow(COLUMNS)
    with open(imed, mode='rt', encoding="utf8") as f:
        data=f.read().replace('\n', '')
        data = re.sub(' +', '', data)
        sentences = re.findall('.*?[！。？]', data)
        for i, s in enumerate(sentences):
            if i == 0: # resolving a bug with start of text character
                s = s[1:]
            for index, char in enumerate(s):
                if char in ['0', '1']:
                    # TODO: I need to clean this
                    # if index < 7:
                    #     B4 = "/s"
                    #     POSB4 = "NAN"
                    # else:
                    #     B4 = s[index - 7]
                    #     _, POSB4= next(pseg.cut(B4))
                    # if index < 5:
                    #     B3 = "/s"
                    #     POSB3 = "NAN"
                    # else:
                    #     B3 = s[index - 5]
                    #     _, POSB3= next(pseg.cut(B3))
                    if index < 3:                          
                        B2 = "/s"
                        POSB2 = "NAN"
                    else:
                        B2 = s[index - 3]
                        _, POSB2= next(pseg.cut(B2))
                    if index == 0:
                        B1 = "/s"
                        POSB1 = "NAN"
                    else:
                        B1 = s[index - 1]
                        _, POSB1= next(pseg.cut(B1))
                    if index == (len(s) - 1):
                        F1 = "/s"
                        POSF1 = "NAN"
                    else:
                        F1 = s[index + 1]
                        _, POSF1= next(pseg.cut(F1))
                    if index > (len(s) - 4):
                        F2 = "/s"
                        POSF2 = 'NAN'
                    else:
                        F2 = s[index + 3]
                        _, POSF2= next(pseg.cut(F2))
                    # if index > (len(s) - 6):
                    #     F3 = "/s"
                    #     POSF3 = "NAN"
                    # else:
                    #     F3 = s[index + 5]
                    #     _, POSF3= next(pseg.cut(F3))
                    # if index > (len(s) - 8):
                    #     F4 = "/s"
                    #     POSF4 = "NAN"
                    # else:
                    #     F4 = s[index + 7]
                    #     _, POSF4= next(pseg.cut(F4))
                    w.writerow([char, str(index//2), B2, B1, F1, F2, POSB2, POSB1,
                                POSF1, POSF2])
    write_file.close()
if __name__ == "__main__":
    inp = sys.argv[1] # name of segmented text file (ex. testing/kw_result.txt)
    # filename for intermediate results, for debugging (ex. testing/kw_clean.txt)
    imed = sys.argv[2]
    out = sys.argv[3] # output csv file for features/labels (ex.testing/kw_test.csv)
    dict_file = sys.argv[4] # output filename for char dictionary (ex.testing/kw_dict.txt)
    main(inp, imed, out, dict_file)
