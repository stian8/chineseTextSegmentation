import re
import sys
import os

DIR = 'raw_data/'
def main():
    write_all_text(DIR)           

def write_all_text(text_dir):
    """Function to write all the groupings, each on its own line,
    to a text file per subject."""
    all_txt = []
    subject = None
    for subdir, dirs, files in os.walk(text_dir):
        if files[0] != '.DS_Store': # DS store
            for file in files:
                path = os.path.join(subdir, file)
                with open(path, mode='rt', encoding="utf8") as f:
                    inp = f.readlines()
                    for line in inp:
                        # replacing delineation characters
                        line = re.sub('[:/,.;；：／，。\(\)]',' ', line)
                        line = re.sub(' +',' ', line)
                        all_txt.extend(line.split())
        else:
            subject = dirs[0] # shortcut to extracting subject name
        if subject:
            write_path = 'processed_data/' + subject + '_result.txt'
            write_file = open(write_path,"w+")
            for g in all_txt:
                write_file.write(g + "\n")
            write_file.close()
    return ''.join(all_txt) # not doing anything currently

if __name__ == "__main__":
    #data = sys.argv[1]
    #main(data)
    main()
