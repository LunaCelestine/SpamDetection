# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:43:34 2018

@author: Bradley Dabdoub
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier

#If true, test data is run against the models
test = True

data = pd.read_csv('spambase.data')
X = data.iloc[:, :48]
Y = data.iloc[:, -1]

#Split data into train/test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Create and fit a mulitinomial Naive Bayes classifier to the training set
multiNB = MultinomialNB()
multiNB.fit(X_train, Y_train)
print("Classification rate for MulitnomialNB on training set:", multiNB.score(X_train, Y_train))

#Create and fit and AdaBoost classifier for ensemble learning
ada = AdaBoostClassifier()
ada.fit(X_train, Y_train)
print("Classification rate for AdaBoost ensemble on training set:", ada.score(X_train, Y_train))

#If the test variable is true, then run the test set through the model
if test:
    print("Classification rate for MulitnomialNB on test set:", multiNB.score(X_test, Y_test))
    print("Classification rate for AdaBoost ensemble on test set:", ada.score(X_test, Y_test))
    