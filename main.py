#numeric
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def readFile(path):

    #import data file
    df = pd.read_csv(path)
    #pd.set_option('display.max_columns', None)

    df.columns = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference', ';', '(', '[', '!', '$', '#', 'crla', 'crll', 'crlt', 'spam']

    # Delete irrelevant features based on correlation with the result

    #plt.figure(figsize=(30, 30))
    cor = df.corr()
    #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.show()

    # Correlation with output variable
    cor_target = abs(cor["spam"])


    aux = 0
    for i in df.columns:

        if cor_target[aux] < 0.1 or cor_target[aux] >0.90: #if they are too different or too similar they dont carry info
            df.drop(i, axis=1, inplace=True)
        aux += 1

    #relevant_features = cor_target[cor_target > 0.5]


    '''    #delete columns
    df.drop('george', axis=1, inplace=True)
    df.drop('000', axis=1, inplace=True)
    df.drop('650', axis=1, inplace=True)
    df.drop('415', axis=1, inplace=True)
    df.drop('85', axis=1, inplace=True)
    df.drop('1999', axis=1, inplace=True)'''

    #print(df)

    perc_info = 0.8
    n_info = int((df.shape[1] - 1) * perc_info)
    n_redu = int((df.shape[1] - 1) * (1-perc_info))

    if n_info + n_redu != df.shape[1] - 1:
        diff = df.shape[1] - 1 - (n_info + n_redu)
        n_info += diff

    print(n_info)
    print(n_redu)


    # define dataset
    X, y = make_classification(n_samples= int(len(df.index) * 0.8), n_features=df.shape[1] -1, n_informative=n_info, n_redundant=n_redu, random_state=3)
    # define the model
    model = RandomForestClassifier()

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    '''    # fit the model on the whole dataset
    model.fit(X, y)
    # make a single prediction
    row = [
        [0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278
]]
    yhat = model.predict(row)
    print('Prediction: %d' % yhat[0])'''



    #boxPlotgraph(df)
    clean_df = df.dropna() #df without rows with NA values (in this case, there are NA values on the table)

    #print(clean_df.corr(method='pearson')) #calculates the correlation (end of 2nd point)

    return clean_df, (mean(n_scores))



def boxPlotgraph(df):


    list = [None] * 58
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    count = 1
    for i in list:
        sns.distplot(df['c' + str(count)+ ''], color='r', ax=axes[0])
        sns.boxplot(df['c' + str(count)+ ''], color='b', ax=axes[1])
        axes[0].set_title('Distribution of Frequence of word "you"')
        axes[1].set_title('Range of Frequence of word "you"')
        plt.show()
        #plt.savefig('new' + str(count) + '.pdf')
        count += 1




def split_analysis(dataframe, param, position_column):

    if (position_column-1 < len(dataframe.columns)): #if the position given is valid (is in the first row)
        print(param)

        listHeader = dataframe.columns.tolist()
        nameColumn = listHeader[position_column-1] #name of the column to compare

        newDF = dataframe.loc[dataframe[nameColumn] == param] #creates a new dataframe filtered by the parameter passed as argument

        print("Number of patients")
        print(len(newDF))

        print(newDF.corr(method='pearson')) #calculates correlation

        return newDF

def analysisOfEveryoneBut (dataframe, param, position_column):
    if (position_column - 1 < len(dataframe.columns)):  # if the position given is valid (is in the first row)
        print("Everyone but " + param)

        listHeader = dataframe.columns.tolist()
        nameColumn = listHeader[position_column - 1]  # name of the column to compare

        newDF = dataframe.loc[dataframe[nameColumn] != param]  # creates a new dataframe filtered by the parameter passed as argument

        print("Number of patients")
        print(len(newDF))

        print(newDF.corr(method='pearson'))  # calculates correlation

        return newDF

def createGraph (general, df1, df2): #3rd point

    features_mean = list(general.columns[2:3])
    features_mean += list(general.columns[4:14])
    print(features_mean)

    numberColumns = features_mean.__len__()

    dfM = df1
    # print(dfM)
    dfB = df2

    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(nrows=numberColumns, ncols=numberColumns, figsize=(15, 15))
    # fig, axes = plt.subplots(nrows=d, ncols=d, sharex=True, sharey=True)
    for i in range(numberColumns):
        for j in range(numberColumns):
            ax = axes[i, j]
            ax.figure
            if i == j:
                ax.text(0.5, 0.5, "Diagonal", transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12)
            else:
                ax.scatter(dfM[features_mean[j]], dfM[features_mean[i]], marker='s', color='r', label='M')
                # hold(True)
                ax.scatter(dfB[features_mean[j]], dfB[features_mean[i]], marker='o', color='g', label='B')

    plt.savefig("new.pdf")
    plt.show()



# Main
if __name__ == '__main__':

    nr_repetitions = 5
    acuracy_avr = 0

    for i in range(0, 5):

        generalDataFrame, avr = readFile("spambase.data")

        acuracy_avr += avr

    acuracy_avr = acuracy_avr / nr_repetitions

    print("acuracy avr - ", acuracy_avr)

    #df1 = split_analysis(generalDataFrame, "0=Blood Donor", 2)
    #df2 = analysisOfEveryoneBut(generalDataFrame, "0=Blood Donor", 2)

    #createGraph(generalDataFrame, df1, df2)


    #missing the SelectKbest part


