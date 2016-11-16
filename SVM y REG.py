# coding=utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import time
from sklearn.feature_extraction import DictVectorizer

#-------------------------------------Training set 0

#Set working directory
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/train/pos')
#Looking at .txt in directory
train_pos = []
list_of_files = glob.glob('./*.txt')
for fileName in list_of_files:
    data_list = open(fileName, 'rb').read()
    train_pos.append(data_list)    
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/test/pos')
#Looking at .txt in directory
list_of_files = glob.glob('./*.txt')
for fileName in list_of_files:
    data_list = open(fileName, 'rb').read()
    train_pos.append(data_list) 
#Set working directory
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/train/neg')
#Looking at .txt in directory
train_neg = []
list_of_files = glob.glob('./*.txt')
for fileName in list_of_files:
    data_list = open(fileName, 'rb').read()
    train_neg.append(data_list)
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/test/neg')
list_of_files = glob.glob('./*.txt')
for fileName in list_of_files:
    data_list = open(fileName, 'rb').read()
    train_neg.append(data_list)
   
#---------------------------------------- Intercalar
text_data = [None] * (len(train_pos) + len(train_neg))
text_data[0::2] = train_neg #posiciones pares
text_data[1::2] = train_pos #posiciones impares

#--------------------------------------------
y=np.tile([0, 1], 1500)
y_train = y[:2000]
y_test = y[2000:]

tf_idf_vectorizer = TfidfVectorizer()
X = tf_idf_vectorizer.fit_transform(text_data)
X_train = X[:2000, :]
X_test = X[2000:, :]

svm = SVC()
logistic_regression = LogisticRegression()
classifiers = {
	"Support vector machine": svm,
	"Logistic regression": logistic_regression
}
performances = {}
time_0 = time.time()
n_folds = 10
print ("Evaluating models")
#--------------------------------------Entrenamiento
for name, classifier in classifiers.items():
	performance = []
	fold_indices_generator = KFold(X_train.shape[0], n_folds=n_folds)
	for (train, test) in fold_indices_generator:
		classifier.fit(X_train[train], y_train[train])
		performance.append(classifier.score(X_train[test], y_train[test]))
	performances[name] = np.mean(performance)
	print ("%s: %s" % (name, performances[name]))
time_f = time.time()
print('time')
print(time_f - time_0)   

max_performance = max(performances.values())
best_classifier_name = [name for (name, classifier) in classifiers.items() if performances[name] == max_performance][0] #Con esto elegimos el clasificador con la mayor performance
print ("Winner: %s" % best_classifier_name)
print ("Performace on test set: %s" % classifiers[best_classifier_name].score(X_test, y_test))


#------------------------Función para contar TOP 10
def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[0])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))