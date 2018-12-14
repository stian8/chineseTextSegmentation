# Naive Bayes Model for Chinese Word Segmentation
import sys
import re, operator
from functools import reduce
import Levenshtein as Lev # imported by pip install python-Levenshtein

# Important: input the clean version of the text from helper

def main(input_file, test_file, output_file):
    with open(input_file, mode='rt', encoding="utf8") as f:
        data= f.read().replace('\n', '')
        data = re.sub(' +', '', data)
    frequency_dict, total_tokens = create_frequency_dictionary(data)
    prob_dict = Pdist(frequency_dict, total_tokens)
    test_text = segment(test_file, output_file, prob_dict)

def create_frequency_dictionary(data):
    """Creates a frequency dictionary of tokens from the text.
    Args:
        data: training corpus with no line breaks or spaces.
    Returns:
        tuple of:
            frequency_dict: tokens to counts
            total_tokens: total number of tokens
    """
    total_tokens = 0
    frequency_dict = {}
    sentences = re.findall('.*?[！。？]', data)
    for i, s in enumerate(sentences):
        if i == 0: # resolving a bug with start of text character
            s = s[1:]
        current = ''
        # building frequency dictionary of tokens
        for char in s:
            if char == '1' and len(current) > 0:
                frequency_dict[current] = frequency_dict.get(current, 0) + 1
                current = ''
                total_tokens += 1
            elif char != '0':
                current += char
    return frequency_dict, total_tokens

def segment(test_file, output_file, prob_dict):
    """Return a text of words that includes the best segmentation for
    the text in the text_file. Returns it with 1 as separated and 0 as
    non-separated. Driver function of recursive_seg. Also writes result to
    the output_file.
    Args:
        test_file: filename of file that contains corpus to test.
        output_file: filename to write result to.
        prob_dict: probability distribution dictionary of training corpus.
    Returns:
        predicted: test text labeled with predicted splits.        
    """
    write_file = open(output_file, mode="w", encoding='utf-8-sig')
    with open(test_file, mode='rt', encoding="utf8") as f:
        # cleaning up text
        data=f.read().replace('\n', '')
        data = re.sub(' +', '', data)
        actual = str(data)
        data = re.sub('0', '', data)
        data = re.sub('1', '', data)
    sentences = re.findall('.*?[！。？]', data) # splitting into sentences
    tokens = []
    # get predicted tokens for each sentence
    for s in sentences:
        result = recursive_seg(s, prob_dict)
        tokens.extend(result)
    intermediate = '1'.join(tokens) # join the split tokens with 1
    final = ['1']
    punc = set('！。？，：；""')
    for char in intermediate:
        if char == '1':
            final[-1] = '1'
        elif char in punc:
            # Fixing a recurisve model parsing issue (?),
            # not what we want to penalize for
            final[-1] = '1'
            final.append(char)
            final.append('1')
        else:
            final.append(char)
            final.append('0')
    predicted = ''.join(final)
    write_file.write(predicted)
    write_file.close()
    print(get_accuracy(actual, predicted))
    return predicted

def memoize(f):
    """Function decorator that provides memoization for the function.
    Credit to python-course.eu for a tutorial about this.
    Args:
        f: function to memoize
    Returns:
        Reference to helper, wraps f in something that saves calculated results
            in f.
    """
    table = {}
    def helper(*args):
            if args[0] not in table:
                    table[args[0]] = f(*args)
            return table[args[0]]
    helper.memo = table
    return helper

@memoize
def recursive_seg(text, prob_dict):
    """Recursive function for finding best splits of the text.
    Generate first, remainder splits to recursively find the best tokenization
    The candidate with the highest probability for the text is selected."""
    combos = [(text[:i+1], text[i+1:]) for i in range(min(len(text), 2))]
    # get split candidates for the text
    raw_cand = [[first]+recursive_seg(rem, prob_dict) for first,rem in combos]
    candidates = [x for x in raw_cand if x!=[''] and x != []]
    try:
        # find highest probabilistic candidate based on priors
        thing = max(candidates, key=lambda x: reduce(operator.mul, [prob_dict(w) for w in x], 1))
    except ValueError:
        return []
    return thing

def get_accuracy(original, predicted):
    """Gets accuracy between the original and predicted text."""
    num_splits = float(original.count('1') + original.count('0'))
    # used Lev distance in case we have a shift, since all the characters
    # should be the same this still gives a good accuracy measure
    return (num_splits - Lev.distance(original, predicted))/num_splits

class Pdist(dict):
    """Probability distribution class formed estimated from the
    inputted frequency dictionary of the training data.
    Credit for idea from the internets."""
    def __init__(self, data, total):
            for key, count in data.items():
                    self[key] = self.get(key, 0) + count
            self.total = float(total)
            self.missingfn = lambda k, total: 1./total
    def __call__(self, key): 
            if key in self: return self[key]/self.total 
            else: return self.missingfn(key, self.total)

if __name__ == '__main__':
    input_file = sys.argv[1] # 0, 1 demarcated training text, ex. training/truth_clean.txt
    test_file = sys.argv[2] # 0, 1 demarcated testing text, ex. testing/kw/kw_clean.txt
    output_file = sys.argv[3] # file to output to, ex. outputs/kw/kw_bayes.txt
    main(input_file, test_file, output_file)
