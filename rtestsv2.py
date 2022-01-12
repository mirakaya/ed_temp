import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the class
from sklearn.linear_model import LogisticRegression
# Splitter
from sklearn.model_selection import train_test_split
# import the metrics class
from sklearn import metrics
#randomgenerator
import random as rd


def init():
    df1 = pd.read_csv('spambase.data',header=None)
    df1.columns=["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", "word_freq_over", "word_freq_remove",\
                                    "word_freq_internet","word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report",\
                                    "word_freq_addresses", "word_freq_free", "word_freq_business", "word_freq_email","word_freq_you", "word_freq_credit", "word_freq_your", \
                                    "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",\
                                    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",\
                                    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",\
                                    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$",\
                                    "char_freq_#", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", "Binary_Spam"]
    #print(df1.head())

    feature_cols = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", "word_freq_over", "word_freq_remove",\
                                    "word_freq_internet","word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report",\
                                    "word_freq_addresses", "word_freq_free", "word_freq_business", "word_freq_email","word_freq_you", "word_freq_credit", "word_freq_your", \
                                    "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",\
                                    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",\
                                    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",\
                                    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$",\
                                    "char_freq_#", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
    X=df1[feature_cols] #features
    Y=df1.Binary_Spam
    #print(X)
    #print(Y)
    return X,Y

def calc(X,Y,testsize=0.1,rs=0):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize, random_state=rs)



    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(tol=(1*10**-8),max_iter=2000) #1977???

    # fit the model with data
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    
    acc=metrics.accuracy_score(y_test, y_pred)
    prc=metrics.precision_score(y_test, y_pred)
    rec=metrics.recall_score(y_test, y_pred)
    print("Accuracy:",acc)
    print("Precision:",prc)
    print("Recall:",rec,"\n")
    
    
    return y_test,y_pred,logreg,X_test,acc,prc,rec
    
def showconf(y_test,y_pred):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #cnf_matrix


    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

def showROC(logreg,X_test,y_test):

    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':

    nr_repetitions = 3
    accuracy_avr = 0
    precision_avr = 0
    recall_avr=0
    X,Y=init()
    tsize=0.1
    rand=0
    
    
    for i in range(0, nr_repetitions):
        print("Test size ",tsize*100,"%\n")
        for j in range(0, nr_repetitions):
            rand=rd.randint(0,2**32-1)
            avr,prc,rec =calc(X,Y,tsize,rand)[-3:]

            accuracy_avr += avr
            precision_avr += prc
            recall_avr+=rec
        
        tsize+=0.1  #10% 20% 30%

    accuracy_avr = accuracy_avr / nr_repetitions**2
    precision_avr = precision_avr / nr_repetitions**2
    recall_avr = recall_avr / nr_repetitions**2
    print("Accuracy avr - ", accuracy_avr)
    print("Precision avr - ", precision_avr)
    print("Recall avr - ", recall_avr)