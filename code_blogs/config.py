#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:44:09 2021

@author: theogady
"""

from nltk.corpus import stopwords

import spacy
#nlp = spacy.load('fr_core_news_sm',disable=['parser', 'ner'])
nlp = spacy.load('fr_core_news_sm')


### Liste des stops words français ###
# french_stopwords = set(stopwords.words('french'))

french_stopwords = nlp.Defaults.stop_words
