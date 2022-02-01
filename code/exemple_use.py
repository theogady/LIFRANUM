#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:20:25 2022

@author: theogady
"""

######################################## Packages ########################################
import pandas as pd

from os import mkdir
##

from other.usefull_tools import divide_data, extract_text_from_html, cleaning, extract_language, limit_X
from topics_approach.Topic_Model_functions import creation_lda_tools, choose_nb_topics, create_lda, load_lda
from clustering_tools.clustering import naive_topics_clustering
from clustering_tools.clusters_analysis import ini_info_cluster, get_most_sallient_words, get_ngram, get_ngram2
from clustering_tools.clusters_analysis import get_most_repr_text, get_size_cluster, labeling, get_style, visu_style, get_extrem_style

######################################## Executed code ########################################
if __name__ == "__main__" :
    
######################################## Data creation ########################################
    raw_data = pd.read_pickle("backup/data/raw_data")
    
    # limiting the size of the content to 1000, the median of our corpus
    raw_data_divided = divide_data(raw_data, lim = 1000, save = True)
    
    # extracting the text from html
    readable_data = extract_text_from_html(raw_data_divided, save = True)

######################################## Topics model ########################################
    meaningfull_data = cleaning(readable_data,  lemmatization = False, stemming = False, ngram = False, save = True)
    meaningfull_data = extract_language(meaningfull_data, save = True)
    data_posts = meaningfull_data[(meaningfull_data.language == "FRENCH") & (meaningfull_data.kind == "blogger#post")]
    
    # limiting the number of posts per author to 200, the median of our corpus
    data_posts_lim = limit_X(data_posts,200, col = "author.displayName")
    
    # backup info file creation
    # infos = pd.DataFrame(columns=["Description"])
    # infos.to_pickle("backup/models/infos")
    infos = pd.read_pickle("backup/models/infos")

    v = 1
    try:
       mkdir('backup/models/test_numb_' + str(v))
       print("File created")
    except:
        print("File already exists")
    
    # creating the tools we need for the lda model training
    dico, corpus_lim, list_data, sample = creation_lda_tools(data_posts_lim, fraq_data = 1, total_text = True)
    # visualisation of the coherence depending on the nb of topics, can be lon
    nb_topics, _ = choose_nb_topics (dico, corpus_lim, list_data, start = 20 , limit = 40, step=2, model = "Mallet")
    # lda model created and saved
    lda_model, vis = create_lda(corpus_lim, dico, nb_topics, list_data,  v = v, model = "Mallet", coherence = False)
    # registers the parameters info
    infos.loc[v] = ["Mallet model with " + str(nb_topics) + " topics"]
    infos.to_pickle("backup/models/infos")
    
    # loads the saved lda model
    lda_model, id2word, corpus,vis = load_lda(v)
    
    # clusterisation based on main topic
    classified = naive_topics_clustering(data_posts,lda_model, v, irr_topics = [])
    
    # Analysis of the clusters
    # All the analysis info about the clusters will be saved in the info_clusters file
    ini_info_cluster(v)
    # Getting most sallient words of each cluster
    get_most_sallient_words(v, Lambda = 0.5, save = True, nb_words = 20, below_lim = 5)
    # Getting most sallient ngrams of each cluster
    get_ngram(v, Lambda = 0.5, save = True)
    # Getting most probable ngrams of each cluster
    get_ngram2(v, save = True)
    # Getting most representative texts of each cluster
    get_most_repr_text(v, nb_texts = 1, nb_url = 10, save = True, min_text_size = 500)
    # Getting size of each cluster
    get_size_cluster(v)

    L_labels = ["Topic_" + str(x) for x in range(nb_topics)]
    # Labeling the clusters
    labeling(v, L_labels)

    # Getting style evaluation of each cluster
    get_style(v, test = "t-test", frac = 0.1, save = True, output="nagg")
    # Visualisation of the most descriminant features
    visu_style(v, N_features = 8, test_min = 1e-04, title="Clusters style evaluation")
    # Getting the most extreme stylistic values and for which cluster
    extreme_values = get_extrem_style(v, N_values = 20)
    # Getting the clusters info file
    info_cluster = pd.read_pickle('backup/models/test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
