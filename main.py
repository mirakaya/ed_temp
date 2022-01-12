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

        if cor_target[aux] < 0.2 or cor_target[aux] >0.9: #if they are too different or too similar they dont carry info
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

    #print(df.columns)

    perc_info = 1
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
    model = RandomForestClassifier() #atm ~93.8%

    #model= LogisticRegression() #atm ~85.4%

    model =

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=15, n_repeats=5, random_state=3)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    # fit the model on the whole dataset
    #model.fit(X, y)
    '''
    # make a single prediction
    row = [
        [0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278
]]
    yhat = model.predict(row)
    print('Prediction: %d' % yhat[0])'''


    #---------untested section neural networks

    '''#n_samples= int(len(df.index) * 0.8), n_features=df.shape[1] -1, n_informative=n_info, n_redundant=n_redu, random_state=3)

    X, y = make_blobs(n_samples=len(df.index), centers=3, n_features=df.shape[1] -1, cluster_std=2, random_state=2)

    # split into train and test
    n_train = int(len(df.index) * 0.8)
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))'''

    # ---------untested section logistic regression

    # Logistic regression model



    '''y = df.iloc[:,-1:]
    X = df


    logreg = LogisticRegression(max_iter=100000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

    rfe = RFE(logreg)  # running RFE with 15 variables as output
    rfe = rfe.fit(X_train, y_train)

    list(zip(X_train.columns, rfe.support_, rfe.ranking_))

    col = X_train.columns[rfe.support_]

    X_train_sm = sm.add_constant(X_train[col])
    logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
    res = logm2.fit()
    res.summary()'''








    #-----------



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

    nr_repetitions = 10
    acuracy_avr = 0

    for i in range(0, nr_repetitions):

        generalDataFrame, avr = readFile("spambase.data")

        acuracy_avr += avr

    acuracy_avr = acuracy_avr / nr_repetitions

    print("acuracy avr - ", acuracy_avr)

    #df1 = split_analysis(generalDataFrame, "0=Blood Donor", 2)
    #df2 = analysisOfEveryoneBut(generalDataFrame, "0=Blood Donor", 2)

    #createGraph(generalDataFrame, df1, df2)


    #missing the SelectKbest part


