# File for cleaning and putting data into correct format
import re
import sys
import jieba.posseg as pseg
import csv

#INPUT = "truth_result.txt"
#INTERMEDIATE = "truth_clean.txt"
#OUTPUT  = "training.csv"
PUNC = set(['，',',','.','!',':',';','“', '"', '，', '。',
            '、', '！', '；', '？','……', '?', '：', '”'])
CON = set(['的','地', '得']) # what to do about le, bu
def main(inp, imed, out):
    clean_text(inp, imed)
    create_csv(imed, out)

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
            elif char == "/":
                final[-1] = '1'
            elif char == "不":
                final.append(char)
                final.append('1')
            elif char == "了":
                final.append(char)
                final.append('1')
            else:
                final.append(char)
                final.append('0')
        write_file.write(''.join(final))
    write_file.close()

def create_csv(imed, out):
    write_file = open(out, mode="w", encoding='utf-8-sig')
    w = csv.writer(write_file)
    window = [None for _ in range(5)]
    # No being which number split it is
    w.writerow(["Label", "No", "B2", "B1", "F1", "F2", "POSB2",
               "POSB1", "POSF1", "POSF2"])
    with open(imed, mode='rt', encoding="utf8") as f:
        data=f.read().replace('\n', '')
        sentences = re.findall('.*?[！。？]', data)
        for s in sentences:
            for index, char in enumerate(s):
                if char in ['0', '1']:
                    # I need to clean this
                    try:
                        B2 = s[index - 3]
                        _, POSB2= next(pseg.cut(B2))
                    except IndexError:
                        B2 = "/s"
                        POSB2 = "NAN"
                    try:
                        B1 = s[index - 1]
                        _, POSB1= next(pseg.cut(B1))
                    except IndexError:
                        B1 = "/s"
                        POSB1 = "NAN"
                    try:
                        F1 = s[index + 1]
                        _, POSF1= next(pseg.cut(F1))
                    except IndexError:
                        F1 = "/s"
                        POSF1 = "NAN"
                    try:
                        F2 = s[index + 3]
                        _, POSF2= next(pseg.cut(F2))
                    except IndexError:
                        F2 = "/s"
                        POSF2 = "NAN"
                    w.writerow([char, str(index//2), B2, B1, F1, F2, POSB2, POSB1,
                                POSF1, POSF2])
    write_file.close()
if __name__ == "__main__":
    inp = sys.argv[1]
    imed = sys.argv[2]
    out = sys.argv[3]
    main(inp, imed, out)