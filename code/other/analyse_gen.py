#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:27:56 2021

@author: theogady
"""
######################################## Packages ########################################
import pandas as pd

from bs4 import BeautifulSoup

import pycld2 as cld2


###
from fonction_import.fonctions_utiles import sent_to_words

######################################## Quelques indicateurs ########################################

### Quelques info sur les langues détéctées ###
def languages(data):
    dico =  dict((y, x) for x, y in cld2.LANGUAGES)
    dico["un"] = "undefined"
    dico["no_text"] = "text_format_problem"
    languages = [dico[i] for i in data["language"].unique()]
    no_fr = data.loc[data.language != "fr"] #48 750 au total
    return(languages, no_fr)


def analyse_reseau(raw_data) :
    posts = raw_data.loc[raw_data.kind == "blogger#post"]["author.id"] #110 895 posts
    com = raw_data.loc[raw_data.kind == "blogger#comment"]['author.id'] # 242 945 comments -> totale 353 840
    len(posts.unique()) # 183 auteurs
    len(com.unique()) # 5 528 comentateurs
    inter = list(set(posts).intersection(set(com)))
    pd.DataFrame(inter).to_pickle("inter")
    len(inter) # 156 en commun
    
    com_author = com.loc[raw_data["author.id"].isin(inter) ]
    
    count = 0
    for author in inter :
        com = com_author.loc[com_author["author.id"] == author]["post.id"]
        for c in com :
            if raw_data.loc[c]["author.id"] != author :
                count +=1
                break
    #137 auteurs ont commenté sur textes à eux
    #109 auteurs ont commenté sur textes pas à eux
    com_author_to_author = pd.DataFrame([])
    for author in inter :
        com = com_author.loc[com_author["author.id"] == author]
        for _,c in com.iterrows() :
            author_to = raw_data.loc[c["post.id"]]["author.id"]
            if author_to != author :
                c["author_to"] = author_to
                com_author_to_author = com_author_to_author.append(c)
    com_author_to_author.to_pickle("com_author_to_author")
#9 251 commentaires faits par des auteurs de blogs -> réseaux de relation à faire

def count_images (raw_data) :
    count_images = 0
    count_posts = 0
    content_images = pd.DataFrame([], columns = ["content","nb_images","images"])
    content_images.content = raw_data.loc[raw_data.kind == "blogger#post"].content
    
    for index, row in content_images.iterrows():
        soup = BeautifulSoup(row.content,'html.parser')
        images = soup.find_all("img")
        row.nb_images = len(images)
        count_images += len(images)
        count_posts += 1 if (len(images) > 0) else 0
        row.images = images
    
    count_images # 155 779
    count_posts # 63 077

def count_labels(cleaned_data) :
    cleaned_posts = cleaned_data[(cleaned_data.language == "fr") & (cleaned_data.kind == "blogger#post")]
    l = pd.isna(cleaned_posts.labels)
    labels = cleaned_posts[l.values == False] # 50 090 posts valables avec commentaires
    liste = labels.labels.values.tolist()
    dict_labels = {}
    for doc in liste :
        for tag in doc :
            tag = " ".join(sent_to_words(tag))
            try : dict_labels[tag] +=1
            except : dict_labels[tag] = 1
    l = dict_labels.items()
    l = sorted(l, key=lambda x: (x[1]), reverse=True)
    l[:20]

### analyse des elements de X les plus importants ###
def get_biggest_X(data, X) :
    list_aut = list(set(data[X].values))
    print("Nb "+str(X) +" = "+ str(len(list_aut)))
    count = {}
    for i, post in data.iterrows() :
        author = post[X]
        if author in count.keys():
            count[author] +=1
        else :
            count[author] = 1
    result = pd.DataFrame(columns = [X, "count"])
    result[X], result["count"] = zip(*count.items())
    result = result.sort_values(by="count", ascending=False)
    result.reset_index(inplace = True, drop = True)
    print(result.head(20))
    print(result.tail(20))
    return result
# 170 auteurs ds data_posts pour 186 blogs -> un meme auteur peut avoir pls blogs
    
    
    
    
    
    