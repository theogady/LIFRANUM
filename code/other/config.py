#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:44:09 2021

@author: theogady
"""


import spacy
#nlp = spacy.load('fr_core_news_sm',disable=['parser', 'ner'])
nlp = spacy.load('fr_core_news_sm')


### List of french stopwords ###

french_stopwords = nlp.Defaults.stop_words

path_to_mallet_binary = "topics/Mallet/bin/mallet"