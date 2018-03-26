#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import urllib
import tarfile

def checkfiles():
    """
    Check if the database is downloaded and extracted, else do the necessary"
    """
    # Create a folder named "data" if there isn't already one
    if "data" not in os.listdir("."):
        os.mkdir("data")
    # Check for .tar archive and download if from ai.stanford.edu if needed
    if "aclImdb_v1.tar" not in os.listdir("data"):
        print("Downloading files from ai.stanford.edu")
        testfile = urllib.URLopener()
        testfile.retrieve("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                          "data/aclImdb_v1.tar")
        print("Done")
    # Extract from the archive if not already done
    if "aclImdb" not in os.listdir("data"):
        print("Extracting from archive...")
        tar = tarfile.open(os.path.join("data","aclImdb_v1.tar"), "r")
        tar.extractall("data")
        tar.close()
        print("Done")

def load_reviews(set_name = "train"):
    """
    Load a train of a test set (default train) as a list of strings (the reviews)
    and a list of labels (0 for negative review, 1 for positive)
    """
    checkfiles() # Check if files are here, download them if not
    if set_name not in ["train","test"]:
        raise ValueError("Name of the set to load must be 'test' or 'train'")
    link = os.path.join("data","aclImdb",set_name)
    X,y = [], []
    for name,label in zip(["neg","pos"], [0, 1]):
        toload = os.path.join(link,name)
        for txt in os.listdir(toload):
            with open(os.path.join(toload,txt), 'r') as f:
                X.append(f.read())
                y.append(label)
    print("{} reviews loaded for {} set".format(len(X), set_name))
    return X,y