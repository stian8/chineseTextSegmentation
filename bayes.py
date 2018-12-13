import sys
import re, string, random, glob, operator, heapq
from math import log10

# input the clean version of the text from helper

def main(input_file, test_file, output_file):
    with open(input_file, mode='rt', encoding="utf8") as f:
        data= f.read().replace('\n', '')
        data = re.sub(' +', '', data)
    frequency_dict, total_tokens = create_frequency_dictionary(input_file)
    prob_dict = Pdist(frequency_dict, total_tokens)
    test_text = segment(test_file, output_file, prob_dict)

def create_frequency_dictionary(data):
    """Creates a frequency dictionary of tokens from the text.
    Args:
        data:
    Returns:
        tuple of:
            frequency_dict: tokens to counts
            total_tokens: total number of tokens
    """
    total_tokens = 0
    frequency_dict = {}
##    with open(input_file, mode='rt', encoding="utf8") as f:
##        data=f.read().replace('\n', '')
##        data = re.sub(' +', '', data)
    sentences = re.findall('.*?[！。？]', data)
    for i, s in enumerate(sentences):
        if i == 0: # resolving a bug with start of text character
            s = s[1:]
        current = ''
        for char in s:
            if char == '1' and len(current) > 0:
                frequency_dict[current] = frequency_dict.get(key, 0) + 1
                current = ''
                total_tokens += 1
            elif char != '0':
                current += char
    return frequency_dict, total_tokens                   

def segment(test_file, output_file, prob_dict):
    "Return a list of words that is the best segmentation of text."
    write_file = open(output_file, mode="w", encoding='utf-8-sig')
    with open(input_file, mode='rt', encoding="utf8") as f:
        data=f.read().replace('\n', '')
        data = re.sub(' +', '', data)
        data = re.sub('0', '', data)
        data = re.sub('1', '', data)
    sentences = re.findall('.*?[！。？]', data)
    tokens = []
    for i, s in enumerate(sentences):
        if i == 0: # resolving a bug with start of text character
            s = s[1:]
        result = recursive_seg(s, prob_dict)
        tokens.extend(result)
    intermediate = "'1'".join(result)
    final = ['1']
    for char in final:
        if char == '1':
            final[-1] = '1'
        else:
            final.append(char)
            final.append('0')
    predicted = ''.join(final)
    write_file.write(predicted)
    return predicted

def recursive_seg(text, prob_dict):
    combos = [(text[:i+1], text[i+1:]) 
			for i in range(len(text))]
    candidates = [[first]+recursive_seg(rem, prob_dict) for first,rem in combos]
    thing = []
    try:
        thing = max(candidates, key=lambda x: product(prob_dict(w) for w in x))
    except:
        pass
    return thing

def get_accuracy(original, predicted):
    num_splits = float(original.count('1') + original.count('0'))
    return sum(o != p for o,p in zip(original, predicted))/num_splits

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data, total):
            for key, count in data.items():
                    self[key] = self.get(key, 0) + count
            self.total = float(total)
            self.missingfn = lambda k, total: 1./total
    def __call__(self, key): 
            if key in self: return self[key]/self.total 
            else: return self.missingfn(key, self.total)

if __name__ == '__main__':
    input_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    main(input_file, test_file, output_file)
