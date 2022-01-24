#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:22:55 2021

@author: theogady
"""
import pandas as pd

from nltk.tokenize import sent_tokenize

import numpy as np


import gensim
import gensim.corpora as corpora

from itertools import compress

import tqdm

import math

###
from fonction_import.style_functions import RemoveSpecialCHs, create_feature
from fonction_import.style_graph import style_spyder_charts, graph_format2, map_features
from fonction_import.fonctions_utiles import freq

######################################## Analyse par cluster ########################################

### Analyse des mots les plus représentatifs du cluster ###
def get_most_sallient_words (v, Lambda = 0.5, save = True, nb_words = 20) :
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    try :
        info_clusters.drop("Most_sallient_words", inplace = True, axis = 1)
    except :
        pass
    df_words = pd.DataFrame(columns = ["Cluster_nb","Most_sallient_words"])
    try :
        lda_model = gensim.models.ldamodel.LdaModel.load('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + '.lda.model')
        df_words = pd.DataFrame(columns = ["Cluster_nb","Most_sallient_words"])
        print("Start evluation most sallient words with lda")
        for i,worlds in tqdm.tqdm(lda_model.show_topics(num_topics=-1,num_words=nb_words, formatted=False)):
            L_worlds = [l[0] for l in worlds]
            df_words.loc[i] = [i,L_worlds]
    except :
        print("Start evluation most sallient words without lda")
        data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
        texts = data.content.values.tolist()
        for i in tqdm.tqdm(list(info_clusters.Cluster_nb)) :
            posts_cluster_i = data[data.Class == i].content.values.tolist()
            words_i = corpora.Dictionary(posts_cluster_i)
            words_i.filter_extremes(no_below=5)
            words_i = words_i.token2id.keys()
            freq_i = freq(words_i,posts_cluster_i)
            total_freq_i = freq(words_i, texts)
            l = [(word, Lambda*math.log(freq_i[word]) + (1-Lambda)*math.log(freq_i[word]/total_freq_i[word])) for word in freq_i.keys()]
            l = sorted(l, key=lambda x: (x[1]), reverse= True)
            l = l[:nb_words]
            try :
                l = [i[0] for i in l]
            except :
                l =[]
            print("\n")
            print(l)
            df_words.loc[i] = [i,l]
    info_clusters = info_clusters.merge(df_words, on = "Cluster_nb")
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

def get_most_sallient_words2 (v, Lambda = 0.5, save = True, nb_words = 20, below_lim = 5) :
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    try :
        info_clusters.drop("Most_sallient_words", inplace = True, axis = 1)
    except :
        pass
    df_words = pd.DataFrame(columns = ["Cluster_nb","Most_sallient_words"])

    print("Start evluation most sallient words without lda")
    data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
    texts = data.content.values.tolist()
    dico_tot = corpora.Dictionary(texts)
    n_tot = dico_tot.num_pos
    freq_tot = dico_tot.cfs
    for i in tqdm.tqdm(list(info_clusters.Cluster_nb)) :
        posts_cluster_i = data[data.Class == i].content.values.tolist()
        corpus = [dico_tot.doc2bow(text, allow_update=False, return_missing=False) for text in posts_cluster_i]
        freq_i={}
        n_i = 0
        for j in corpus:
            for item,count in dict(j).items():
                if item in freq_i:
                    freq_i[item]+=count
                else:
                    freq_i[item] = count
                n_i += count
        #freq_i.filter_extremes(no_below=below_lim)
        l = [(dico_tot[word], Lambda*math.log(freq_i[word]/n_i) + (1-Lambda)*math.log((freq_i[word]*n_tot)/(freq_tot[word]*n_i))) for word in freq_i.keys()]
        l = sorted(l, key=lambda x: (x[1]), reverse= True)
        l = l[:nb_words]
        # try :
        #     l = [i[0] for i in l]
        # except :
        #     l =[]
        print("\n")
        print(l)
        df_words.loc[i] = [i,l]
    info_clusters = info_clusters.merge(df_words, on = "Cluster_nb")
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)


def get_ngram (v, Lambda =0.5, save = False, nb_words = 20):
    # Mettre num cluster en foction taille comme sur représentation graph
    # On peut donner des mots de liaison mais ils ont été enlevés comme stopwords
    print("Initialisation")
    data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
    texts = data.content.values.tolist()
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=10.0)
    trigram = gensim.models.Phrases(bigram[texts], threshold=10.0)
    
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    texts_tri = []
    
    print("\n Creation of n-grams")
    for doc in tqdm.tqdm(texts) :
        set_uni_word = set(doc)
        doc_tri = trigram_mod[bigram_mod[doc]]
        texts_tri.append([word for word in doc_tri if word not in set_uni_word ])
    data["Trig"] = texts_tri
    dico_tot = corpora.Dictionary(texts_tri)
    n_tot = dico_tot.num_pos
    freq_tot = dico_tot.cfs
    
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    try :
        info_clusters.drop("Most_sallient_Big/Trig", inplace = True, axis = 1)
    except :
        pass
    df_big = pd.DataFrame(columns = ["Cluster_nb","Most_sallient_Big/Trig"])
    print("Start evaluation ngrams")
    for i in tqdm.tqdm(list(info_clusters.Cluster_nb)) :
        posts_cluster_i = data[data.Class == i].Trig.values.tolist()
        corpus = [dico_tot.doc2bow(text, allow_update=False, return_missing=False) for text in posts_cluster_i]
        freq_i={}
        n_i = 0
        for j in corpus:
            for item,count in dict(j).items():
                if item in freq_i:
                    freq_i[item]+=count
                else:
                    freq_i[item] = count
                n_i += count
        l = [(dico_tot[word], Lambda*math.log(freq_i[word]/n_i) + (1-Lambda)*math.log((freq_i[word]*n_tot)/(freq_tot[word]*n_i))) for word in freq_i.keys()]
        l = sorted(l, key=lambda x: (x[1]), reverse=True)
        l = l[:nb_words]
        # try :
        #     l = [i[0] for i in l]
        # except :
        #     l =[]
        print("\n")
        print(l)
        df_big.loc[i] = [i,l]
    info_clusters = info_clusters.merge(df_big, on = "Cluster_nb")
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

def get_ngram2 (v, save = True, nb_words = 20) :
    # Mettre num cluster en foction taille comme sur représentation graph
    # On peut donner des mots de liaison mais ils ont été enlevés comme stopwords
    print("Calculating the n-grams")
    data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
    texts = data.content.values.tolist()
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=10.0)
    trigram = gensim.models.Phrases(bigram[texts], threshold=10.0)
    
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    try :
        info_clusters.drop("Most_sallient_Big/Trig", inplace = True, axis = 1)
    except :
        pass
    df_big = pd.DataFrame(columns = ["Cluster_nb","Most_sallient_Big/Trig"])
    print("Start evaluation ngrams")
    for i in tqdm.tqdm(list(info_clusters.Cluster_nb)) :
        posts_cluster_i = data[data.Class == i].content.values.tolist()
        big_dict = bigram_mod.find_phrases(posts_cluster_i)
        trig_dict = trigram_mod.find_phrases(posts_cluster_i)
        most_sallient_big = sorted(big_dict.keys(), key= lambda x : big_dict[x])[:int(nb_words/2)]
        most_sallient_trig = sorted([x for x in trig_dict.keys() if x not in most_sallient_big], key= lambda v : trig_dict[v])[:int(nb_words/2)]
        l = most_sallient_trig + ["/"] + most_sallient_big
        print("\n")
        print(l)
        df_big.loc[i] = [i,l]
    info_clusters = info_clusters.merge(df_big, on = "Cluster_nb")
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

def get_most_repr_text(v, save = True) :
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    readable_data = pd.read_pickle("Svg_data/Readable_Data")
    try :
        info_clusters.drop("Text", inplace = True, axis = 1)
    except :
        pass
    data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
    groups = data.groupby('Class')
    most_texts = pd.DataFrame(columns = ["Cluster_nb","Text"])
    for i, grp in groups:
        id_text = grp.sort_values(['Proba'], ascending=[0]).head(1).index
        l = readable_data.loc[str(id_text.values[0])].content
        most_texts.loc[i] = [i, l]
        
    info_clusters = info_clusters.merge(most_texts, on="Cluster_nb")
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

def get_most_repr_text2(v, nb_url = 5, save = True) :
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    readable_data = pd.read_pickle("Svg_data/Readable_Data")
    try :
        info_clusters.drop(["Text","Url"], inplace = True, axis = 1)
    except :
        pass
    data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
    groups = data.groupby('Class')
    most_texts = pd.DataFrame(columns = ["Cluster_nb","Text","Url"])
    print("Start evaluation most repr texts")
    for i, grp in tqdm.tqdm(groups):
        grp.Proba = grp.apply(lambda x: x.Proba * min(500,len(x.content)), axis = 1)
        id_text = grp.sort_values(['Proba'], ascending=[0]).head(nb_url).index
        l = readable_data.loc[str(id_text.values[0])].content
        u= readable_data.loc[id_text.values].url.values
        most_texts.loc[i] = [i, l, u]
        
    info_clusters = info_clusters.merge(most_texts, on="Cluster_nb")
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

def ini_info_cluster(v, save = True):
    info_clusters = pd.DataFrame()
    data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
    info_clusters["Cluster_nb"] = sorted(data.Class.unique())
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

# On ne fait que compter le nombre de doc par cluster, très grossier
# possibilité d'avoir la Marginal topic distribution comme sur graph ?
def get_size_cluster(v, save = True):
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
    n = len(data.index)
    try :
        info_clusters.drop("Perc_size", inplace = True, axis = 1)
    except :
        pass
    df_sizes = pd.DataFrame(columns = ["Cluster_nb","Perc_size"])
    print("Start evaluation of sizes")
    for i in tqdm.tqdm(info_clusters.Cluster_nb) :
        sub = data.loc[data.Class == i]
        df_sizes.loc[i] = [i, round(len(sub.index)/n,4)*100]
    info_clusters = info_clusters.merge(df_sizes, on="Cluster_nb")
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

# Un label "ESPACE" n'est pas pris en compte pcq je ne sais pas ce que c'est
# débat sur le pretraitement
### Sauvegarde de la moyenne de chaque feature dans "info_clusters" ###
###+ fichier total avec mean/std/p-value pour chaque feature ###
def get_style(v, test = "t-test", frac = 0.01, save = True, output="nagg"):
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    readable_data = pd.read_pickle("Svg_data/Readable_Data")
    
    l_drop = ['Function words', 'Indexes', 'Letters', 'NER',
       'Numbers', 'Punctuation', 'Structural', 'TAG',"cluster"]
    l_drop = l_drop + list(map_features.keys())
    for i in l_drop :
        try :
            info_clusters.drop([i], inplace = True, axis = 1)
        except :
            pass
    data = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".Classified_Posts")
    stylo_dict = {}
    
    print("Start evaluation of features")
    sample = readable_data.loc[data.index].sample(frac = frac, axis = 0, random_state=2)
    for index , row in tqdm.tqdm(sample.iterrows()) :
        text = row["content"]
        sent_text = sent_tokenize(text)
        words=RemoveSpecialCHs(text)
        stylo_dict[(data.loc[index].Class, index)]=create_feature(text, words, sent_text)
    
    stylo_df=pd.DataFrame(stylo_dict).transpose().rename_axis(['cluster', 'id']).reset_index().fillna(0)
    results = graph_format2(stylo_df, test, output)
    
    results_mean = results[results.index=='mean']
    results_mean.set_index("cluster", inplace = True)
    
    for cluster in info_clusters.Cluster_nb.unique() :
        if cluster not in results_mean.index :
            results_mean.loc[cluster] = 0

    info_clusters = info_clusters.merge(results_mean, left_on="Cluster_nb", right_index= True)
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
        results.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".style_eval")
    return info_clusters

### Permet la visualistaion des "N_features" features les plus discrimants ###
def visu_style(v, N_features = 8):
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    try : results = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".style_eval")
    except : print("No style_eval file available")
    
    results_test = results[results.index=='test']

    N_tot_features = len(results.columns) -1

    test_list = np.array(results_test.drop(['cluster'], axis=1)).ravel()
    min_indices = iter(test_list.argsort())
    feature_indices = []
    while len(feature_indices) != N_features :
        f = next(min_indices)
        f = f%N_tot_features
        if f not in feature_indices :
            feature_indices.append(f)
            
    results_to_graph = results.iloc[:,feature_indices]
    results_to_graph["cluster"] = results.cluster

    try :
        results_to_graph.cluster = results_to_graph.apply(lambda x : info_clusters.loc[info_clusters.Cluster_nb == x.cluster].Label.values[0], axis = 1)
    except :
        pass
    style_spyder_charts(results_to_graph, N_features)
    return info_clusters

### Permet obtenir les "N_values" valeurs les plus extrèmes
def get_extrem_style(v, N_values = 20):
    try : results = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".style_eval")
    except : print("No style_eval file available")
    
    results_test = results[results.index=='test']
    N_tot_features = len(results.columns) -1

    test_list = np.array(results_test.drop(['cluster'], axis=1)).ravel()
    min_indices = iter(test_list.argsort())
    values = []
    while len(values) != N_values :
        i = next(min_indices)
        f = i%N_tot_features
        c = i//N_tot_features
        value_serie = results.iloc[3*c:3*c+3,f]
        dict1 = {"Cluster" : c, "Feature" : value_serie.name}
        dict2 = dict(value_serie)
        values.append({**dict1, **dict2})
    return values

### Permet labelisation des clusters selon "L_labels"
# Ne peut on pas remplacer numéro cluster par le label pr le retrouver partout après ?
def labelisation(v,L_labels, save = True):
    info_clusters = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    try :
        info_clusters.drop("Label", inplace = True, axis = 1)
    except :
        pass
    info_clusters["Label"] = info_clusters.apply(lambda x : L_labels[x.Cluster_nb], axis = 1)
    if save :
        info_clusters.to_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

def compare_classification(data1, data2):
    clusters = set(data1.Class.values)
    dict_12 = {}
    for i, row1 in data1.iterrows():
        C1 = row1.Class
        C2 = data2.loc[i]
        if C1 not in dict_12.keys() :
            return 0

def dist(dic12, l1, l2) :
    l1bis = [dic12[x] for x in l1]
    s = 0
    for i in range(len(l1)) :
        if l1bis[i] != l2[i] :
            s+=1
    return s

import random as r
import itertools
import numpy as np
from collections import OrderedDict
import time

# on part du principe que l1 et l2 ont la meme taille, et leurs set aussi
def hill_climbing(l1, l2, lim  = 50 ) :
    permu = list(set(l1))
    r.shuffle(permu)
    best_dic12 = OrderedDict(zip(permu, set(l2)))
    best_dist = dist(best_dic12, l1,l2)
    it = 0
    while it < lim :
        it +=1
        print(it)
        print(best_dist)
        combi = list(itertools.combinations(permu, 2))
        list_dist = []
        for c in combi :
            dic12 = best_dic12.copy()
            dic12[c[0]], dic12[c[1]] = dic12.pop(c[1]), dic12.pop(c[0])
            list_dist.append(dist(dic12, l1, l2))
        d = min(list_dist)
        if d >= best_dist or d == 0:
            break
        best_dist = d
        c = combi[np.argmin(list_dist)]
        best_dic12[c[0]], best_dic12[c[1]] = best_dic12.pop(c[1]), best_dic12.pop(c[0])
    return best_dic12, best_dist

def gene(l1,l2, lim = 50, population = 10) :
    permu = list(set(l1))
    l_children = []
    #for i in range(population*(population-1)*2 + population) :
    for i in range(population*2) :
        r.shuffle(permu)
        l_children.append(permu.copy())
    it = 0
    while it < lim :
        it+= 1
        l_distances = []
        for child in l_children :
            dic12 = dict(zip(child,set(l2)))
            l_distances.append(dist(dic12,l1,l2))
        sort = sorted(zip(l_children, l_distances), key = lambda y : y[1])
        adults = [x[0] for x in sort[:population]]
        best_dist = sort[0][1]
        print(it)
        print(best_dist)
        if best_dist == 0 :
            break
        l_children = children2(adults)
    best_dic12 = OrderedDict(zip(adults[0], set(l2)))
    return best_dic12, best_dist

def gene2(l1,l2, popu_ini,  n_adults, lim = 50) :
    l_children = popu_ini
    it = 0
    while it < lim :
        it+= 1
        l_distances = []
        for child in l_children :
            dic12 = dict(zip(child,set(l2)))
            l_distances.append(dist(dic12,l1,l2))
        sort = sorted(zip(l_children, l_distances), key = lambda y : y[1])
        adults = [x[0] for x in sort[:n_adults]]
        best_dist = sort[0][1]
        print(it)
        print(best_dist)
        if best_dist == 0 :
            break
        l_children = children2(adults)
    return adults[0], best_dist

def RandR (l1,l2,lim = 50, n_adults = 10) :
    center = list(set(l1))
    it = 0
    n = len(center)
    best_dist = dist(dict(zip(center,set(l2))),l1,l2)
    while it < lim and n > 0:
        it+= 1
        popu_ini = []
        for i in range(n_adults*2):
            popu_ini.append(n_transpo(center, n))
        new_center, new_dist = gene2(l1,l2, popu_ini, n_adults, lim = 10)
        if new_dist < best_dist :
            n = n/2
        else :
            n = n*2
        center, best_dist = new_center, new_dist
    best_dic12 = OrderedDict(zip(center, set(l2)))
    return best_dic12, best_dist

def n_transpo(l, n):
    for i in range(n) :
        j = r.randint(0, len(l)-1)
        k = len(l)%(j+1)
        l[j], l[k] = l[k], l[j]
    return l

def children (adults):
    l_children = []
    for couple in itertools.combinations(adults, 2) :
        l_children += fusion(couple[0], couple[1])
        l_children += fusion(couple[1], couple[0])
    for adult in adults :
        l_children.append(modify(adult, n= int(len(adult)/2)))
    return l_children

def children2 (adults):
    l_children = []
    for i in range(int(len(adults)/2)) :
        l_children += fusion(adults[2*i], adults[2*i + 1])
        l_children += fusion(adults[2*i + 1], adults[2*i])
    for adult in adults :
        l_children.append(modify(adult, n= int(len(adult)/2)))
    return l_children


def modify(l1, n) :
    l = l1.copy()
    indices_to_change = r.sample(range(len(l1)),n)
    new_indices = indices_to_change.copy()
    r.shuffle(new_indices)
    for i in range(len(indices_to_change)) :
        l[indices_to_change[i]] = l1[new_indices[i]]
    return l


def fusion(l1, l2) :
    g = l1[:int(len(l1)/2)]
    while len(g) < len(l1) :
        i = l2[len(g)]
        if i not in g :
            g.append(i)
        else :
            g.append(None)
    missing = set(l1) - set(g)
    while len(missing) > 0 :
        g[g.index(None)] = missing.pop()
        
    d = l1[int(len(l1)/2):]
    while len(d) < len(l1) :
        i = l2[-len(d)-1]
        if i not in d :
            d.insert(0,i)
        else :
            d.insert(0,None)
    missing = set(l1) - set(d)
    while len(missing) > 0 :
        d[d.index(None)] = missing.pop()
    return [g,d]









