#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:22:55 2021

@author: theogady
"""
import pandas as pd


import numpy as np


import gensim
import gensim.corpora as corpora


import tqdm

import math

###
# from fonction_import.style_functions import RemoveSpecialCHs, create_feature
# from fonction_import.style_graph import style_spyder_charts, graph_format2, map_features
# from other.usefull_tools import freq
from style_approach.style_graph import map_features, graph_format, style_spyder_charts
from style_approach.fr_extractor import sent_tokenize, RemoveSpecialCHs, create_feature
######################################## Clusters analysis ########################################
def ini_info_cluster(v, save = True):
    info_clusters = pd.DataFrame()
    data = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".classified_posts")
    info_clusters["cluster_nb"] = sorted(data.cluster.unique())
    if save :
        info_clusters.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

### Getting the 20 most sallient words of each cluster based on the following expression : ###
### salliency(word, cluster_i) = lambda * log(freq(word in cluster_i)) + (1-lambda) * log(freq(word in cluster_i)/freq(word all corpus)) ###
def get_most_sallient_words (v, Lambda = 0.5, save = True, nb_words = 20, below_lim = 5) :
    info_clusters = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    if "most_sallient_words" in  info_clusters.columns.tolist():
        info_clusters.drop("most_sallient_words", inplace = True, axis = 1)
    df_words = pd.DataFrame(columns = ["cluster_nb","most_sallient_words"])

    print("Start evluation most sallient words")
    data = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".classified_posts")
    meaningfull_data = pd.read_pickle('backup/data/meaningfull_data')
    data.content = meaningfull_data.loc[data.index].content
    texts = data.content.values.tolist()
    dico_tot = corpora.Dictionary(texts)
    dico_tot.filter_extremes(no_below = below_lim)
    n_tot = dico_tot.num_pos
    freq_tot = dico_tot.cfs
    for i in tqdm.tqdm(list(info_clusters.cluster_nb)) :
        posts_cluster_i = data[data.cluster == i].content.values.tolist()
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
        l = sorted(l, key=lambda x: (x[1]), reverse= True)
        l = l[:nb_words]
        try :
            l = [i[0] for i in l]
        except :
            l =[]
        print("\n")
        print(l)
        df_words.loc[i] = [i,l]
    info_clusters = info_clusters.merge(df_words, on = "cluster_nb")
    if save :
        info_clusters.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

### Getting the most sallient ngrams, the same way as get_most_sallient_words ###
def get_ngram (v, Lambda =0.5, save = False, nb_words = 20):
    print("Initialization")
    data = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".classified_posts")
    meaningfull_data = pd.read_pickle('backup/data/meaningfull_data')
    data.content = meaningfull_data.loc[data.index].content
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
    
    info_clusters = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    if "most_sallient_ngram" in  info_clusters.columns.tolist():
        info_clusters.drop("most_sallient_ngram", inplace = True, axis = 1)

    df_big = pd.DataFrame(columns = ["cluster_nb","most_sallient_ngram"])
    print("Start evaluation ngrams")
    for i in tqdm.tqdm(list(info_clusters.cluster_nb)) :
        posts_cluster_i = data[data.cluster == i].Trig.values.tolist()
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
        try :
            l = [i[0] for i in l]
        except :
            l =[]
        print("\n")
        print(l)
        df_big.loc[i] = [i,l]
    info_clusters = info_clusters.merge(df_big, on = "cluster_nb")
    if save :
        info_clusters.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return(info_clusters)

### Getting the n-grams of each cluster with highest score from gensim.models.Phraser ###
def get_ngram2 (v, save = True, nb_words = 20) :
    print("Calculating the n-grams")
    data = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".classified_posts")
    meaningfull_data = pd.read_pickle('backup/data/meaningfull_data')
    data.content = meaningfull_data.loc[data.index].content
    texts = data.content.values.tolist()
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=10.0)
    trigram = gensim.models.Phrases(bigram[texts], threshold=10.0)
    
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    info_clusters = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    if "most_sallient_ngram" in  info_clusters.columns.tolist():
        info_clusters.drop("most_sallient_ngram", inplace = True, axis = 1)
        
    df_big = pd.DataFrame(columns = ["cluster_nb","most_sallient_ngram"])
    print("Start evaluation ngrams")
    for i in tqdm.tqdm(list(info_clusters.cluster_nb)) :
        posts_cluster_i = data[data.cluster == i].content.values.tolist()
        big_dict = bigram_mod.find_phrases(posts_cluster_i)
        trig_dict = trigram_mod.find_phrases(posts_cluster_i)
        most_sallient_trig = sorted(trig_dict.keys(), key= lambda v : trig_dict[v])[:int(nb_words/2)]
        most_sallient_big = sorted([x for x in trig_dict.keys() if x not in most_sallient_trig], key= lambda x : big_dict[x])[:nb_words-len(most_sallient_trig)]
        l = most_sallient_trig + most_sallient_big
        print("\n")
        print(l)
        df_big.loc[i] = [i,l]
    info_clusters = info_clusters.merge(df_big, on = "cluster_nb")
    if save :
        info_clusters.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return info_clusters

### Getting the text (as it appears in readable_data) and the urls of the texts with highest "proba" * len(text) value of each corpus ###
### len(text) can be limited if there is a high difference in texts sizes ###
def get_most_repr_text(v, nb_texts = 1, nb_url = 5, save = True, min_text_size = float('inf')) :
    info_clusters = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    readable_data = pd.read_pickle("backup/data/readable_data")
    if any (x in info_clusters.columns.tolist() for x in ["text","urls"]) :
        info_clusters.drop(["text","urls"], inplace = True, axis = 1)
    data = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".classified_posts")
    groups = data.groupby('cluster')
    most_texts = pd.DataFrame(columns = ["cluster_nb","text","urls"])
    print("Start evaluation most repr texts")
    for i, grp in tqdm.tqdm(groups):
        grp.proba = grp.apply(lambda x: x.proba * min(min_text_size,len(x.content)), axis = 1)
        id_text = grp.sort_values(['proba'], ascending=[0]).index
        l = readable_data.loc[id_text.values].head(nb_texts).content.values
        u = readable_data.loc[id_text.values].head(nb_url).url.values
        most_texts.loc[i] = [i, l, u]
        
    info_clusters = info_clusters.merge(most_texts, on="cluster_nb")
    if save :
        info_clusters.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return info_clusters

### Counting the percentage size of each cluster ###
def get_size_cluster(v, save = True):
    info_clusters = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    data = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".classified_posts")
    n = len(data.index)
    if "perc_size" in info_clusters.columns.tolist() :
        info_clusters.drop("perc_size", inplace = True, axis = 1)

    df_sizes = pd.DataFrame(columns = ["cluster_nb","perc_size"])
    print("Start evaluation of sizes")
    for i in tqdm.tqdm(info_clusters.cluster_nb) :
        sub = data.loc[data.cluster == i]
        df_sizes.loc[i] = [i, round(len(sub.index)/n,4)*100]
    info_clusters = info_clusters.merge(df_sizes, on="cluster_nb")
    if save :
        info_clusters.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return info_clusters

### Allows labeling of the clusters, that will be used instead of clusters number ###
def labeling(v,L_labels, save = True):
    info_clusters = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    if "label" in info_clusters.columns.tolist() :
        info_clusters.drop("label", inplace = True, axis = 1)
    info_clusters["label"] = info_clusters.apply(lambda x : L_labels[x.cluster_nb], axis = 1)
    if save :
        info_clusters.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    return info_clusters

### Evaluation of the style features on a fraction of the corpus ###
### Results are given as the mean/stdv for each cluster, are normalized and can be aggregated ###
### We also compute statistical test, either "t-test" or "ks-test" ###
def get_style(v, test = "t-test", frac = 0.01, save = True, output="nagg"):
    info_clusters = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    readable_data = pd.read_pickle("backup/data/readable_data")
    
    l_drop = ['Function words', 'Indexes', 'Letters', 'NER', 'Numbers', 'Punctuation', 'Structural', 'TAG',"cluster"]
    l_drop = l_drop + list(map_features.keys())
    for i in l_drop :
        if i in info_clusters.columns.tolist() :
            info_clusters.drop([i], inplace = True, axis = 1)
    data = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".classified_posts")
    stylo_dict = {}
    
    print("Starting evaluation of features")
    sample = readable_data.loc[data.index].sample(frac = frac, axis = 0, random_state=2)
    for index , row in tqdm.tqdm(sample.iterrows()) :
        text = row["content"]
        sent_text = sent_tokenize(text, space=True)
        text = ''.join(sent_text)
        words=RemoveSpecialCHs(text)
        stylo_dict[(data.loc[index].cluster, index)]=create_feature(text, words, sent_text)
    
    stylo_df=pd.DataFrame(stylo_dict).transpose().rename_axis(['cluster', 'id']).reset_index().fillna(0)
    results = graph_format(stylo_df, test, output)
    
    results_mean = results[results.index=='mean']
    results_mean.set_index("cluster", inplace = True)
    
    for cluster in info_clusters.cluster_nb.unique() :
        if cluster not in results_mean.index :
            results_mean.loc[cluster] = 0

    info_clusters = info_clusters.merge(results_mean, left_on="cluster_nb", right_index= True)
    if save :
        info_clusters.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
        results.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".style_eval")
    return info_clusters

### Allows the visualisation of the N_features most statistically discriminant ###
def visu_style(v, N_features = 8, test_min = 1e-04, title="Clusters style evaluation"):
    info_clusters = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
    try : results = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".style_eval")
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

    if "label" in info_clusters.columns :
        results_to_graph.cluster = results_to_graph.apply(lambda x : info_clusters.loc[info_clusters.cluster_nb == x.cluster].label.values[0], axis = 1)

    style_spyder_charts(results_to_graph, N_features, test_min = 1e-04, title="Clusters style evaluation")
    return info_clusters

### Gets "N_values" with highest "test" values ###
def get_extrem_style(v, N_values = 20):
    try : results = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".style_eval")
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
        dict1 = {"cluster" : c, "feature" : value_serie.name}
        dict2 = dict(value_serie)
        values.append({**dict1, **dict2})
    return values








