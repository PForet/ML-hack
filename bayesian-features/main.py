#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from IMDB import load_reviews
from text_processing import string_to_vec
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from NBSVC import NBSVM
from sklearn.model_selection import train_test_split

# Load the training and testing sets
train_set, y_train = load_reviews("train")
test_set, y_test = load_reviews("test")

# Transform the training and testing sets
myvectorizer = string_to_vec(train_set, method="TFIDF")
X_train = myvectorizer.transform(train_set)
X_test = myvectorizer.transform(test_set)

clf = NBSVM(alpha=0.1,C=12)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))