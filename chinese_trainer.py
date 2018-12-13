# Logistic Regression Model for Chinese 
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import metrics
import csv
import json
import codecs
import re

LABEL = "Label"
COLUMNS = ["Label", "No", "B2", "B1", "F1", "F2", "POSB2", "POSB1",
           "POSF1", "POSF2"]
POS_DICT = training_dict = json.load(open('pos.txt', 'r'))

def main(training, testing, tr_dict, te_dict, output_file):
    training_data = pd.read_csv(open(training), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    testing_data = pd.read_csv(open(testing), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    training_dict = json.load(codecs.open(tr_dict, 'r', 'utf-8-sig'))
    testing_dict = json.load(codecs.open(te_dict, 'r', 'utf-8-sig'))
    corpus_dict = merge_dictionaries(training_dict, testing_dict)
    reverse_dict = reverse_dictionary(corpus_dict)
    format_data(training_data, corpus_dict)
    format_data(testing_data, corpus_dict)
    model = generate_model(training_data)
    test_nolabel = testing_data.drop(LABEL,axis=1)
    actual = testing_data[LABEL]
    outputs = generate_predictions(model, test_nolabel)
    reconstruct_data(outputs, output_file, reverse_dict)
    print(metrics.accuracy_score(actual, outputs[LABEL]))

def format_data(data, corpus_dict):
    """Helper function to format the features in the input data frame
        into a form that trained on. All characters are given IDs, while
        POS tags are label encoded.
    """
    data.drop(["No"], axis = 1)
    # data["B4"] = data["B4"].apply(lambda x: corpus_dict[x])
    # data["B3"] = data["B3"].apply(lambda x: corpus_dict[x])
    data["B2"] = data["B2"].apply(lambda x: corpus_dict[x])
    data["B1"] = data["B1"].apply(lambda x: corpus_dict[x])
    data["F1"] = data["F1"].apply(lambda x: corpus_dict[x])
    data["F2"] = data["F2"].apply(lambda x: corpus_dict[x])
    # data["F3"] = data["F3"].fillna('NAN')
    # data["F3"] = data["F3"].apply(lambda x: corpus_dict[x])
    # data["F4"] = data["F4"].fillna("NAN")
    # data["F4"] = data["F4"].apply(lambda x: corpus_dict[x])
    # data["POSB4"] = data["POSB4"].apply(lambda x: POS_DICT[x])
    # data["POSB3"] = data["POSB3"].apply(lambda x: POS_DICT[x])
    data["POSB2"] = data["POSB2"].apply(lambda x: POS_DICT[x])
    data["POSB1"] = data["POSB1"].apply(lambda x: POS_DICT[x])
    data["POSF1"] = data["POSF1"].apply(lambda x: POS_DICT[x])
    data["POSF2"] = data["POSF2"].fillna("NAN")
    data["POSF2"] = data["POSF2"].apply(lambda x: POS_DICT[x])
    # data["POSF3"] = data["POSF3"].fillna("NAN")
    # data["POSF3"] = data["POSF3"].apply(lambda x: POS_DICT[x])
    # data["POSF4"] = data["POSF4"].fillna("NAN")
    # data["POSF4"] = data["POSF4"].apply(lambda x: POS_DICT[x])

##    data["POSB3"] = data["POSB3"].astype('category')
##    data["POSB3"] = data["POSB3"].cat.codes
##    data["POSB2"] = data["POSB2"].astype('category')
##    data["POSB2"] = data["POSB2"].cat.codes
##    data["POSB1"] = data["POSB1"].astype('category')
##    data["POSB1"] = data["POSB1"].cat.codes
##    data["POSF1"] = data["POSF1"].astype('category')
##    data["POSF1"] = data["POSF1"].cat.codes
##    data["POSF2"] = data["POSF2"].astype('category')
##    data["POSF2"] = data["POSF2"].cat.codes
##    data["POSF3"] = data["POSF3"].astype('category')
##    data["POSF3"] = data["POSF3"].cat.codes

def merge_dictionaries(training_dict, testing_dict):
    """Merges dictionaries of character -> ID such that each character
        has its own unique ID.
    Args:
        training_dict: dictionary of characters -> ID in training data
        testing_dict: dictionary of characters -> ID in testing data,
            the IDs do not necessarily have to be the same/different as
            the training_dict.
    Returns:
        A merged dictionary giving each character in both the training and
        testing corpus a unique ID.
    """
    dictionary = training_dict
    index = max(training_dict.values()) + 1
    for key in testing_dict.keys():
        if key not in dictionary:
                dictionary[key] = index
                index += 1
    dictionary['/s'] = index
    return dictionary

def reverse_dictionary(dictionary):
    """Function to reverse a dictionary from keys:values to values:keys.
    Args:
        dictionary: A dictionary.
    Returns:
        The reverse of the input dictionary.
    """
    new_dictionary = {}
    for key, value in dictionary.items():
        new_dictionary[value] = key
    return new_dictionary
    
def generate_model(data):
    """Creates and trains our model, which is used to basic Chinese word segmentation.
    Args:
        data: A pandas dataframe containing our segmentation training data.
    Returns:
        A trained logistic regression model. 
    """
    # X contains all features except the label
    X = data.drop(LABEL,axis=1)
    # Y contains the label
    Y = data[LABEL]
    # Splitting the data in training and testing data
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
    # Creating the model
    LR = LogisticRegression(penalty='l2')
    # Fitting the model
    LR.fit(X_train,Y_train)
    # Generating predictions
    predictions=LR.predict(X_test)
    print(LR.score(X_test,Y_test))
    # When uncommented, this will let us see the list of predictions
    # print('True values:', Y_test.tolist())
    # print('Predictions:', predictions.tolist())
    # Metrics
    predict_proba = LR.predict_proba(X_test)
    accuracy_score = metrics.accuracy_score(Y_test, predictions)
    print('Accuracy: %.2f' % (accuracy_score))
    print(len(LR.coef_[0]))
    print(LR.coef_[0])
    #classification_report = metrics.classification_report(Y_test, predictions)
    #print('Report:', classification_report)
    return LR

def generate_predictions(model, testing):
    """Generates predictions for the inputted data using the model.
    Args:
        model: A trained logistic regression model. 
        testing: Chinese text of a different corpus, without the labels.
    Returns:
        input text with predictinos, prints results to "Segmentations.csv". 
    """
    predictions=model.predict(testing)
    testing[LABEL] = predictions
    testing.to_csv("Segmentations.csv")
    return testing

def reconstruct_data(outputs, output_file, reverse_dict):
    """Reconstructs the data from the predicted data using a dictionary
        mapping ID to Chinese characters. Writes result to the output filename.
    Args:
        outputs: pandas dataframe contained the predicted labels, along with
            the original features.
        output_file: filename of the file to write the results to.
        reverse_dictionary: a dictionary mapping ID to Chinese characters.
    Returns:
        None
    """
    result = []
    for index, row in outputs.iterrows():
        char = row["B1"]
        label = row["Label"]
        if label == 1: # a split
            result.append(reverse_dict[char])
            result.append('/')
        else: # label is 0 and there is no split
            result.append(reverse_dict[char])
        # we have reached a sentence end
        if reverse_dict[row["F2"]] == '/s':
            result.append(reverse_dict[row["F1"]])
    # remove all the filler strings
    str_res = re.sub('/s', '', ''.join(result))
    # making sure no double // added
    str_res = re.sub('//', '/', str_res)
    with open(output_file, mode='w', encoding='utf-8-sig') as file:
        file.write(str_res)

if __name__ == "__main__":
    training = sys.argv[1]
    testing = sys.argv[2]
    tr_dict = sys.argv[3]
    te_dict = sys.argv[4]
    output_file = sys.argv[5]
    main(training, testing, tr_dict, te_dict, output_file)
