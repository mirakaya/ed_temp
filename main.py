#numeric
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.datasets import make_blobs


import statsmodels.api as sm



# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# graphics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from IPython.display import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#other library for graphs
import seaborn as sns


def readFile(path): #reads the file

    #import data file
    df = pd.read_csv(path)
    df.columns = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference', ';', '(', '[', '!', '$', '#', 'crla', 'crll', 'crlt', 'spam']

    df = df.dropna()  # df without rows with NA values (in this case, there are NA values on the table)

    return df

def randomForest(df, perc_info):


    n_info = int((df.shape[1] - 1) * perc_info)
    n_redu = int((df.shape[1] - 1) * (1 - perc_info))

    if n_info + n_redu != df.shape[1] - 1:
        diff = df.shape[1] - 1 - (n_info + n_redu)
        n_info += diff

    print(n_info)
    print(n_redu)

    # define dataset
    X, y = make_classification(n_samples=int(len(df.index) * 0.8), n_features=df.shape[1] - 1, n_informative=n_info,
                               n_redundant=n_redu, random_state=3)
    # define the model
    model = RandomForestClassifier()

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=15, n_repeats=5, random_state=3)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    return mean(n_scores)


def linearRegression(df, perc_info):

    n_info = int((df.shape[1] - 1) * perc_info)
    n_redu = int((df.shape[1] - 1) * (1 - perc_info))

    if n_info + n_redu != df.shape[1] - 1:
        diff = df.shape[1] - 1 - (n_info + n_redu)
        n_info += diff

    print(n_info)
    print(n_redu)

    #define dataset
    X, y = make_classification(n_samples=int(len(df.index) * 0.8), n_features=df.shape[1] - 1, n_informative=n_info,
                               n_redundant=n_redu, random_state=3)
    #define the model
    model = LogisticRegression()

    #evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=15, n_repeats=5, random_state=3)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    #report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    return mean(n_scores)

def df_treatment(correlationMin, correlationMax):

    # Delete irrelevant features based on correlation with the result
    cor = df.corr() #calculate correlations

    #Correlation graph
    #plt.figure(figsize=(30, 30))
    #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.show()

    # Correlation with output variable
    cor_target = abs(cor["spam"])

    aux = 0
    for i in df.columns:

        if cor_target[aux] < correlationMin or cor_target[aux] > correlationMax: #if they are too different or too similar they dont carry info
            df.drop(i, axis=1, inplace=True)
        aux += 1


# Main
if __name__ == '__main__':

    nr_repetitions = 3
    accuracy_avr = 0

    df = readFile("spambase.data")
    #df_treatment(0.2, 0.9) #if we're using randomforest

    df_treatment(0.1, 0.9) #if we're using linear regression
    print(df.shape)
    for i in range(0, nr_repetitions):

        #avr = randomForest(df, 1) #~93.8% accuracy
        avr = linearRegression(df, 1) #~88.5% accuracy

        accuracy_avr += avr

    accuracy_avr = accuracy_avr / nr_repetitions

    print("Accuracy avr - ", accuracy_avr)


