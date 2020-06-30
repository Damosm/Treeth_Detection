"""
Created on Sun Jun 28 07:52:50 2020

@author: MonOrdiPro
"""

from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

df1 = pd.read_csv('tooth_train_df.csv')
df1 = df1.sample(frac=1).reset_index(drop=True)

#%%

X = df1.iloc[:,:-1]
y = df1.iloc[:,-1:]   


X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)
#%%

parameters = {'alpha' : [.001, .01],
              'max_iter' : [10000, 30000],
              'tol' : [ .0001],
              'loss' : ['hinge', 'log', 'modified_huber'],
              'n_jobs' : [-1]}
sgd = SGDClassifier()

clf = GridSearchCV(sgd, parameters)


clf.fit(X, y.values.ravel())


print(clf.cv_results_)
print('##'*30)
print(sorted(clf.cv_results_.keys()))
print('##'*30)
print(clf.best_estimator_)
print('##'*30)
print(clf.best_estimator_.score(X, y))    



dump(clf.best_estimator_, 'svm.joblib') 