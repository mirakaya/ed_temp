from sklearn.neural_network import MLPClassifier
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# import the class
from sklearn.linear_model import LogisticRegression
# Splitter
from sklearn.model_selection import train_test_split
# import the metrics class
from sklearn import metrics

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


# Delete irrelevant features based on correlation with the result
cor = df1.corr() #calculate correlations

#Correlation graph
'''plt.figure(figsize=(30, 30))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()'''

# Correlation with output variable
cor_target = abs(cor["Binary_Spam"])

aux = 0
for i in df1.columns:

    if cor_target[aux] < 0.11 or cor_target[aux] > 0.9: #if they are too different or too similar they dont carry info

        if i != "Binary_Spam":
            df1.drop(i, axis=1, inplace=True)

    aux += 1

dfaux = df1.drop("Binary_Spam", axis = 1, inplace=False)

feature_cols = dfaux.columns



X=df1[feature_cols] #features
Y=df1.Binary_Spam

#Valores do relatório Iterations = 2000, lr = 0.3, mc = 0.2

#Random_state foi dado por mim, para obter um resultado mais proximo do relatório
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=13)
clf = MLPClassifier(learning_rate='constant', random_state=0, momentum=0.2, learning_rate_init=0.3, max_iter=2000).fit(X_train, y_train)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))