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
    data["B2"] = data["B2"].apply(lambda x: corpus_dict[x])
    data["B1"] = data["B1"].apply(lambda x: corpus_dict[x])
    data["F1"] = data["F1"].apply(lambda x: corpus_dict[x])
    data["F2"] = data["F2"].apply(lambda x: corpus_dict[x])
    data["POSB2"] = data["POSB2"].astype('category')
    data["POSB2"] = data["POSB2"].cat.codes
    data["POSB1"] = data["POSB1"].astype('category')
    data["POSB1"] = data["POSB1"].cat.codes
    data["POSF1"] = data["POSF1"].astype('category')
    data["POSF1"] = data["POSF1"].cat.codes
    data["POSF2"] = data["POSF2"].astype('category')
    data["POSF2"] = data["POSF2"].cat.codes

def merge_dictionaries(training_dict, testing_dict):
    dictionary = training_dict
    index = max(training_dict.values()) + 1
    for key in testing_dict.keys():
        if key not in dictionary:
                dictionary[key] = index
                index += 1
    dictionary['/s'] = index
    return dictionary

def reverse_dictionary(dictionary):
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
    LR = LogisticRegression(penalty='l2', C=100)
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
        testing: Chinese text of a different corpus, without the label.
    Returns:
        input text with predictinos, prints results to "Segmentations.csv". 
    """
    predictions=model.predict(testing)
    testing[LABEL] = predictions
    testing.to_csv("Segmentations.csv")
    return testing

def reconstruct_data(outputs, output_file, reverse_dict):
    result = []
    for index, row in outputs.iterrows():
        char = row["B1"]
        label = row["Label"]
        if label == 1:
            result.append(reverse_dict[char])
            result.append('/')
        else:
            result.append(reverse_dict[char])
        if reverse_dict[row["F2"]] == '/s':
            result.append(reverse_dict[row["F1"]])
    str_res = re.sub('/s', '', ''.join(result))
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
