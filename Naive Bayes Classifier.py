#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import time

from textblob.classifiers import NaiveBayesClassifier

#-------------------------------------Training set 0

#Set working directory
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/train/pos')
#Looking at .txt in directory
pos = []
list_of_files = glob.glob('./*.txt')
for fileName in list_of_files:
    data_list = open(fileName, 'rb').read()
    pos.append((data_list, 'pos'))    
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/test/pos')
#Looking at .txt in directory
list_of_files = glob.glob('./*.txt')
for fileName in list_of_files:
    data_list = open(fileName, 'rb').read()
    pos.append((data_list, 'pos')) 
#Set working directory
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/train/neg')
#Looking at .txt in directory
neg = []
list_of_files = glob.glob('./*.txt')
for fileName in list_of_files:
    data_list = open(fileName, 'rb').read()
    neg.append((data_list, 'neg'))
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/test/neg')
list_of_files = glob.glob('./*.txt')
for fileName in list_of_files:
    data_list = open(fileName, 'rb').read()
    neg.append((data_list, 'neg'))

#---------------------------------- Set entrenamiento y test
train = pos[:1000] + neg[:1000]
test = pos[1000:] + neg[1000:]
#--------------------------------Entrenar clasificador
time_0 = time.time()
cl = NaiveBayesClassifier(train)
time_f = time.time()

# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))

# Show 10 most informative features
cl.show_informative_features(10)