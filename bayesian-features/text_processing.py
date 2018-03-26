#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
def tokenize(s):
    """ Custom tokenizer that extract words from strings using regular expressions
    Match all words and some ponctuation
    """
    return re.findall(r"[\w']+|!|@|!!",s)
    
def processing(s):
    """ Preprocessing steps to apply to all strings before tokenization
    Just make sure all letters are lowercase
    """
    return s.lower()

def string_to_vec(X, method="Count", **kwargs):
    """ Scikit CountVectorier or TfidfVectorizer with different default values
    Uses the preprocessor and tokenizer defined above
    Defaults can be overwritten with kwargs
    """
    # Only two supported method
    if method not in ["Count","TF","TFIDF"]:
        raise ValueError('Method must be one of: "Count","TF","TFIDF"')
    # The defaults values we want for scikit Vectorizer
    newkwargs = {"tokenizer":tokenize, # custom tokenizer to keep ponctuation
              "preprocessor":processing, #custom preprocessor
              "ngram_range":(1,2),
              "min_df":6, # minimum number of occurence for a word to be included
              "max_df":0.99 # maximim frequency for a word to be included
              }
    # Overwrite the defaults with kwargs
    for k,val in kwargs.iteritems():
        newkwargs[k] = val
    
    # Use IDF only it it is the chosen method.
    if method in ["TF","TFIDF"]:
        newkwargs["sublinear_tf"] = True #See scikit doc for info
        newkwargs["use_idf"] = method == "TFIDF"

    # Select the write vectorizer
    if method == "Count":
        vectorizer = CountVectorizer
    else:
        vectorizer = TfidfVectorizer
    
    # Fit to the set with our custom arguments 
    return vectorizer(**newkwargs).fit(X)
