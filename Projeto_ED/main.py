#numeric
import numpy as np
import pandas as pd

# graphics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def readFile(path):

    #import data file
    df = pd.read_csv(path)
    print(df)
#    print(df.size) #number of cells

#    print(df.isnull().sum()) #there are empty cells and they are recognized

    clean_df = df.dropna() #df without rows with NA values (in this case, there are NA values on the table)
    #print(len(clean_df))

    print(clean_df.corr(method='pearson')) #calculates the correlation (end of 2nd point)


    #correlation revision
    #Module value of 1 means the two variables have a perfect correlation
    #0 means the two variables have no correlation
    #Negative correlation ex - as the temperature rises, the snowfall decreases
    #Positive correlation ex - as the temperature rises, so does the sale of sunscreen
    #A negative is just as strong as its positive counterpart, the module value is the important thing

    return clean_df

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
    generalDataFrame = readFile("spambase.data")

    df1 = split_analysis(generalDataFrame, "0=Blood Donor", 2)
    df2 = analysisOfEveryoneBut(generalDataFrame, "0=Blood Donor", 2)

    createGraph(generalDataFrame, df1, df2)


    #missing the SelectKbest part


