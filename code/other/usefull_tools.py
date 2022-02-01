#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:10:03 2021

@author: theogady
"""


import json

import pandas as pd
import re
from os import listdir
from os.path import isfile

from nltk.stem.snowball import FrenchStemmer

from bs4 import BeautifulSoup

import pycld2 as cld2

import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import tqdm

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import requests

import time
import numpy as np

###
from other.config import french_stopwords, nlp



######################################## usefull pre processing tools ########################################

def divide_html (raw_text):
    soup = BeautifulSoup(raw_text, 'html.parser')
    return list(soup.stripped_strings)

def resize (raw_text, lim):
    result = []
    string = str()
    for l in divide_html (raw_text) :
        string += " " + l
        if len(string) > lim :
            result.append(string)
            string = str()
    if len(string)>0 :
        result.append(string)
    return result

def divide_data_list_series(row, lim = 500):
    l = []
    l_texts = resize (row.content, lim)
    for text in l_texts:
        new_row = row.copy()
        new_row.content = text
        l.append(new_row)
    return l

### Division of a DataFrame based on the size of its "content" ###
def divide_data (data, lim = 500, save = False):
    copy = data.copy()
    copy = copy.apply(divide_data_list_series, args = [lim], axis = 1)
    copy = copy.apply(pd.DataFrame, args = [])
    copy = pd.concat(copy.values)
    copy = copy.reset_index(drop=True)
    if save :
        copy.to_pickle("backup/data/raw_data_divided")
    return copy

def html_to_text(raw_text):
   ### Adding a space after line break, to avoid words fusion ###
  cleanr = re.compile("<br />")
  cleaned = re.sub(cleanr, '<br /> ', ''.join(raw_text))
  soup = BeautifulSoup(cleaned, 'html.parser')
  text = soup.get_text()
  return (text)

### Text extraction from html in the "content" of a DataFrame ###
def extract_text_from_html(data, save = False):
    copy = data.copy()
    copy["content"] = data["content"].apply(html_to_text, args=[])
    if save :
        copy.to_pickle("backup/data/readable_data")
    return copy

filtre_stopfr =  lambda text: [token for token in text if token.lower() not in french_stopwords]

### Tokenisation + low + removes ponctuation + one letter word + accents + numbers ###
def sent_to_words(sentence):
    return(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(text) 
    texts_out = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return texts_out

def stemming (text):
    stemmer = FrenchStemmer()
    texts_out = [stemmer.stem(word) for word in text]
    return texts_out

### Cleanning the text, keeping only meaningfull words, returns a list ###
def clean_text (raw_text, lemm = False, stemm = False):
    cleanr = re.compile(r'\n|<.*?>|\b[^ \n]*http[^ \n]*\b|\b[^ \n]*\#[^ \n]*\b|\b[^ \n]*\.com\b|’')
    cleaned = re.sub(cleanr, ' ', ''.join(raw_text))
    if lemm :
        cleaned = lemmatization(cleaned)

    cleaned = filtre_stopfr( list(sent_to_words(cleaned)))
    if stemm :
        cleaned = stemming(cleaned)
    return cleaned

### Lemmatization and stemming make the processing very slow ###
def cleaning(data, lemmatization = False, stemming = False, ngram = False, save = False):
    copy = data.copy()
    copy["content"] = data["content"].apply(clean_text, args=[lemmatization,stemming])
    if ngram :
        list_data = copy.content.values.tolist()
        bigram = gensim.models.Phrases(list_data, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[list_data], threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        list_data = [trigram_mod[bigram_mod[doc]] for doc in list_data]
        copy["content"] = list_data
    if save :
        copy.to_pickle("backup/data/meaningfull_data")
    return copy

def getlang(text):
    text = " ".join(text)
    try :
        _,_,d = cld2.detect(text)
        return d[0][0]
    except :
        return "no_text"

### Adds the "language" column on a DataFrame, based on its "content"
def extract_language(data, save = False):
    copy = data.copy()
    copy["language"] = copy.content.apply(getlang, args = [])
    if save :
        copy.to_pickle("backup/data/meaningfull_data")
    return copy

### Limits the number of occurencies of each different value in col ###
def limit_X(data, nb_max, col = "author.displayName"):
    copy = data.copy()[0:0]
    list_X = list(set(data[col].values))
    for l in list_X :
        data_l = data[data[col] == l]
        if len(data_l) > nb_max :
            data_l = data_l.sample(n = nb_max, axis = 0, random_state=1)
        copy = copy.append(data_l)
    return copy

######################################## other usefull tools ########################################

def freq(wordDict, corpus):
    wordDict2 = dict.fromkeys(wordDict, 0)
    total_count = 0
    for text in corpus :
        for world in text :
            if world in wordDict2.keys(): 
                wordDict2[world]+=1
            total_count +=1
    for word, count in wordDict2.items():
            wordDict2[word] = count/total_count
    return (wordDict2)

######################################## usefull tools ########################################




def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in french_stopwords] for doc in texts]


# enlever les \n et \t mtn ou juste pr words ?
def clean_text2 (raw_text):
    cleanr = re.compile(r'<.*?>')
    cleaned = re.sub(cleanr, ' ', ''.join(raw_text))
    return (cleaned)

def remove_links (raw_text):
    soup = BeautifulSoup(raw_text, 'html.parser')
    for atag in soup.find_all('a'):
        atag.extract()
    return str(soup)



### Initialisation de la base de données pour WorldPress ###
def creation_data_W():
    data = pd.DataFrame([])
    blogs_file = "wordpress_blogs/blogs_wordpress_130421.json"
    comments_file = "wordpress_blogs/comments_wordpress_130421.json"
    posts_file = "wordpress_blogs/posts_wordpress_130421.json"
    with open(blogs_file, "r") as r1, open(comments_file, "r") as r2, open(posts_file, "r") as r3:
        JSON_blogs = json.loads(r1.read())
        JSON_comments = json.loads(r2.read())
        JSON_posts = json.loads(r3.read())
    data_blogs = pd.DataFrame(JSON_blogs).transpose()[["ID","name","description","URL","subscribers_count"]]
    data_posts = pd.DataFrame(JSON_posts).transpose()
    data_posts_norm = pd.json_normalize(JSON_posts)
    
    return 0


def getlang(text):
    text = " ".join(text)
    try :
        _,_,d = cld2.detect(text)
        return(d[0][0])
    except :
        return("no_text")











def make_cloud (dictionnary) :
    wc = WordCloud(background_color="white", max_words=100, width=800, height=400)
    wc.generate_from_frequencies(dictionnary)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# marche pas tres bien pr données non numérqiue (par ex sur distrib_X(raw_data, "blog.name"))
def distrib_X(data,X) :
    print(data[X].describe())
    plt.figure()
    #plt.hist(data[X], color = 'blue', edgecolor = 'black', bins = len(set(data[X].values)),  density = False, align = "left")
    plt.hist(data[X], color = 'blue', edgecolor = 'black', bins = "auto",  density = False, align = "mid")
    plt.axvline(x=50, color='r', linestyle='-')
    plt.axvline(x=36, color='g', linestyle='--')
    plt.axvline(x=65, color='g', linestyle='--')
    plt.show()



def test (data, start, stop, nb) :
    tests = np.logspace(start, stop, nb)
    for t in tests :
        t = int(t)
        print("Size : " + str(t))
        t1 = time.time()
        df = divide_data1(data.head(t))
        print ("divide_data1, time = " + str(time.time() - t1))
        
        t1 = time.time()
        df = divide_data2(data.head(t))
        print ("divide_data2, time = " + str(time.time() - t1))
        
        t1 = time.time()
        df = divide_data3(data.head(t))
        print ("divide_data3, time = " + str(time.time() - t1))
    

