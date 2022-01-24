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
from config import french_stopwords, nlp

######################################## Fonctions utiles ########################################

filtre_stopfr =  lambda text: [token for token in text if token.lower() not in french_stopwords]

### Extraction text de html ###
def html_to_text(raw_text):
   ### pour mettre un espace entre les sauts de ligne, pr eviter fusion mots quand pas de point ###
  cleanr = re.compile("<br />")
  cleaned = re.sub(cleanr, '<br /> ', ''.join(raw_text))
  soup = BeautifulSoup(cleaned, 'html.parser')
  text = soup.get_text()
  return (text)

### Tokenisation + low + retire ponctuation + mots d'une lettre + accents + chiffres ###
def sent_to_words(sentence):
    return(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in french_stopwords] for doc in texts]

### Cleanning ###
def clean_text (raw_text):
    ## lemmatisation/stemmatisation cf https://www.datacorner.fr/nltk/
    ## et si le http est en debut de phrase ? essayer \b?
    ## trouver commande pour enlever liens en gen (.fr, .net)
    ### enlever les residus de balises et liens http et # et .com ###
    cleanr = re.compile(r'\n|<.*?>|\b[^ \n]*http[^ \n]*\b|\b[^ \n]*\#[^ \n]*\b|\b[^ \n]*\.com\b|’')
    cleaned = re.sub(cleanr, ' ', ''.join(raw_text))
    cleaned = filtre_stopfr( list(sent_to_words(cleaned)))
    return (cleaned)

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

### Initialisation de la base de données pour Blogger ###
def creation_data():
    dossier = "blogger_blogs"
    data = pd.DataFrame([])
    dossiers = listdir(dossier)
    dossiers.remove(".DS_Store")
    for f in dossiers:
        file = dossier + "/" + f + "/blog_posts_" + f + ".json"
        file2 = dossier + "/" + f + "/blog_info_" + f + ".json"
        file3 = dossier + "/" + f + "/blog_comments_" + f + ".json"
        if (isfile(file)):
            with open(file, "r") as read_file:
                JSON_posts = json.loads(read_file.read())
                DF_posts = pd.json_normalize(JSON_posts, meta = ["autor","blog","replies"])
            if isfile(file2) :
                with open(file2, "r") as read_file:
                    JSON_info = json.loads(read_file.read())
                    DF_info = pd.json_normalize(JSON_info, meta = ["locale","pages","posts"])
                    DF_info = DF_info.add_prefix("blog.")
                DF_posts = pd.merge(DF_posts, DF_info, on = "blog.id")
            data = data.append(DF_posts)
        if (isfile(file3)):
            with open(file3, "r") as read_file:
                JSON_comments = json.loads(read_file.read())
                DF_comments = pd.json_normalize(JSON_comments, meta = ["post","blog","author"])
                if isfile(file2)and not DF_comments.empty:
                    DF_comments = pd.merge(DF_comments, DF_info, on = "blog.id")
            data = data.append(DF_comments)
    data.to_pickle("Data")

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

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def stemmatisation (texts):
    stemmer = FrenchStemmer()
    texts_out = []
    for doc in texts :
        texts_out.append([stemmer.stem(word) for word in doc])
    return (texts_out)

def freq(wordDict, corpus):
    wordDict2 = dict.fromkeys(wordDict, 0)
    total_count = 0
    for text in corpus :
        for world in text :
            try : 
                wordDict2[world]+=1
            except : 
                pass
            total_count +=1
    for word, count in wordDict2.items():
        # if count == 0:
        #     wordDict2[word] = -1
        # else :
            wordDict2[word] = count/total_count
    return (wordDict2)

def limit_X(data, nb_max, col = "author.displayName"):
    copy = data.copy()[0:0]
    list_X = list(set(data[col].values))
    for l in list_X :
        data_l = data[data[col] == l]
        if len(data_l) > nb_max :
            data_l = data_l.sample(n = nb_max, axis = 0, random_state=1)
        copy = copy.append(data_l)
    return copy

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

def divide_data(data, lim = 500):
    d1 = data.loc[data.content.apply(lambda x : len(x) <= lim)]
    d2 = data.loc[data.content.apply(lambda x : len(x) > lim)]
    d3 = pd.DataFrame(columns = d1.columns)
    for i, row in tqdm.tqdm(d2.iterrows()):
        L = resize(row.content, lim)
        for l in L :
            row.content = l
            d3 = d3.append(row)
    return d1.append(d3)

def divide_data_df(row, lim = 500):
    df = pd.DataFrame(columns = row.index)
    l_texts = resize (row.content, lim)
    for text in l_texts:
        new_row = row.copy()
        new_row.content = text
        df = df.append(new_row)
    return df

def divide_data_list_series(row, lim = 500):
    l = []
    l_texts = resize (row.content, lim)
    for text in l_texts:
        new_row = row.copy()
        new_row.content = text
        l.append(new_row)
    return l

def divide_data_list_series2(row, lim = 500):
    l = []
    l_texts = resize (row.content, lim)
    for text in l_texts:
        new_row = row.copy()
        new_row.content = text
        l.append(new_row.content)
    return l

# c'est le divide le plus rapide
def divide_data1 (data, lim = 500):
    copy = data.copy()
    copy = copy.apply(divide_data_list_series, args = [lim], axis = 1)
    copy = copy.apply(pd.DataFrame, args = [])
    copy = pd.concat(copy.values)
    return copy

def divide_data2 (data, lim = 500):
    copy = data.copy()
    copy = copy.apply(divide_data_df, args = [lim], axis = 1)
    copy = pd.concat(copy.values)
    return copy

# semble etre un peu mieux, mais pas bon
def divide_data3 (data, lim = 500):
    copy = data.copy()
    copy = copy.apply(divide_data_list_series, args = [lim], axis = 1)
    copy = copy.apply(pd.DataFrame, args = [])
    copy = pd.concat(copy.values)
    copy = pd.merge (copy, data.drop("content", inplace=False, axis = 1), left_index = True, right_index =  True)
    return copy

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
    

