import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import metrics
import csv

LABEL = "Label"
COLUMNS = ["Label", "No", "B2", "B1", "F1", "F2", "POSB2", "POSB1",
           "POSF1", "POSF2"]

def main(training, testing):
    training_data = pd.read_csv(open(training), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    testing_data = pd.read_csv(open(testing), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    model = generate_model(training_data)
    test_nolabel = testing_data.drop(LABEL,axis=1)
    actual = testing_data[LABEL]
    outputs = generate_predictions(model, test_nolabel)
    print(metrics.accuracy_score(actual, outputs[LABEL]))
    
    
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
    predictions=model.predict(inputs)
    testing[LABEL] = predictions
    testing.to_csv("Segmentations.csv")
    return testing

if __name__ == "__main__":
    training = sys.argv[1]
    testing = sys.argv[2]
    main(training, testing)
