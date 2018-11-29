import re
import sys
def main(file_name):
    with open(file_name, encoding="utf8") as f:
        inp = f.readlines()
        for line in inp:
            line = re.sub('[／，。\(\)]',' ', line)
            line = re.sub(' +',' ', line)
            print(line.split())

if __name__ == "__main__":
    data = sys.argv[1]
    main(data)
