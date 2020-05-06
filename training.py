import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

if __name__ == '__main__':
    corona=pd.read_csv("corona.csv")
    X=corona.drop('infectionProb',axis=1)
    Y=corona['infectionProb'].copy()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=42)
    clf=LogisticRegression()
    clf.fit(X_train,Y_train)
    predictions=clf.predict(X_test)
    print("Accuracy = {0:.3f}".format(metrics.accuracy_score(Y_test, predictions)))

    ## Open a file where you want to store the data
    file=open('corona.pkl','wb')

    ## Dump information to that file
    pickle.dump(clf,file)
    file.close()