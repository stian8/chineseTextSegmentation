import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import metrics
import csv

def main(filename):
    inputs = pd.read_csv(open(filename), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=0, engine="python")
    # we really need think about how we are representing the data
    
    

if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)
