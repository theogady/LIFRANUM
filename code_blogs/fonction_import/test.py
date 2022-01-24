#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:33:37 2021

@author: theogady
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from fonction_import.fonctions_utiles import make_cloud

###
from config import french_stopwords

######################################## Premiere analyse frequences ########################################
### cf https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html ###
def analyse_freq(data) :
    content_data = data["content"]
    if type(data.head(1)["content"].values[0]) == list :
        content_data = data["content"].apply(' '.join, args=[])
    content_data = content_data.to_frame()
    count_vect = CountVectorizer()
    
    X_train_counts = count_vect.fit_transform(content_data.content)
    tfidf_transformer = TfidfTransformer(use_idf=False)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    freq = pd.DataFrame(X_train_tfidf.mean(0).tolist()[0], columns = ["freq"])
    vocab = pd.DataFrame.from_dict(count_vect.vocabulary_,orient='index',columns=['indices'])
    vocab = vocab.sort_values(by = 'indices', ascending=True)
    freq["mot"] = vocab.index
    freq = freq.sort_values(by = "freq", ascending=False)
    freq["test"] = freq.mot.apply(lambda x : x in french_stopwords)
    freq.reset_index(drop = True, inplace = True)
    print(freq.head(30))
    print(freq.tail(30))
    dico = dict(zip(freq.mot, freq.freq))
    make_cloud(dico)
    return(freq)

######################################## Classifier comment/post ########################################
def clasifier_com_post(data):
    content_data = data["content"].apply(' '.join, args=[])
    content_data = content_data.to_frame()
    content_categorie_data = content_data
    content_categorie_data["categorie"] = data.kind
    train_data = content_categorie_data.sample(frac = 0.5, axis = 0, random_state=1)
    test_data = content_categorie_data[~content_data.index.isin(train_data.index)]
    
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data.content)
    tfidf_transformer = TfidfTransformer(use_idf=False)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    clf = MultinomialNB().fit(X_train_tfidf, train_data.categorie)
    
    X_new_counts = count_vect.transform(test_data.content)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    predicted = clf.predict(X_new_tfidf)
    test_data["results"] = predicted
    test_data["results"] = (test_data["results"] == test_data["categorie"])
    
    print(metrics.classification_report(test_data.categorie, predicted))
    print(metrics.confusion_matrix(test_data.categorie, predicted))
    return(test_data)