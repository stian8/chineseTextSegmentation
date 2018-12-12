import re
import sys
import os

#DIR = 'raw_data/'
#OUT = 'testing/'
def main(input_dir, output_dir):
    write_all_text(input_dir, output_dir)           

def write_all_text(text_dir, output_dir):
    """Function to write all the groupings, each on its own line,
    to a text file per subject."""
    subject = None
    for subdir, dirs, files in os.walk(text_dir):
        all_txt = ''
        for file in files:
            if file != '.DS_Store': # DS store
                path = os.path.join(subdir, file)
                with open(path, mode='rt', encoding="utf8") as f:
                    inp = f.readlines()
                    for line in inp:
                        # replacing delineation characters
                        #line = re.sub('[“”、?？！!:/,.;；：／，。\(\)]',' ', line)
                        line = re.sub(' +','', line)
                        all_txt += line
        index = subdir.rfind('/')
        subject = subdir[index+1:]
        if subject:
            # I was thinking that here we could write to csv then load into pandas
            write_path = output_dir + subject + '_result.txt'
            write_file = open(write_path,"w+")
            write_file.write(all_txt)
            #for g in all_txt:
                #write_file.write(g + "\n")
            write_file.close()
    return ''.join(all_txt) # not doing anything currently

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)
