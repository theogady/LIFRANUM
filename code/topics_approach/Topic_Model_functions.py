#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:18:18 2021

@author: theogady
"""

import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel


import pyLDAvis
import pyLDAvis.gensim_models

import matplotlib.pyplot as plt

import pickle

###
from topics_approach import ldamallet
from other.config import path_to_mallet_binary

######################################## Topic Modeling ########################################

### Creating the dictionnary and document/term matrix needed for the lda model ###
def creation_lda_tools(data, fraq_data = 1, total_text = True) :
    ### The vocabulary can be learned on a fraction of the corpus ###
    sample = data.sample(frac = fraq_data, axis = 0, random_state=2)
    list_data = sample.content.values.tolist()
    ### Dictionnary creation ###
    id2word = corpora.Dictionary(list_data)
    ### Filtering rare and extreme vocabulary ###
    id2word.filter_extremes(no_below=5, no_above=0.5)

    ### We can train the model on all the data set or not ###
    if total_text :
        list_data = data.content.values.tolist()
    ### Creating the document/term matrix ###
    corpus = []
    ### Occurencies of a term in a document is limited to 10 ###
    for i in range(len(list_data)) :
        bow_i = id2word.doc2bow(list_data[i])
        bow_i = [[x[0],min(10,x[1])] for x in bow_i]
        corpus.append(bow_i)
    return(id2word, corpus, list_data, sample)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("nb topics : " + str(num_topics))
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
            random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def compute_coherence_values_mallet(dictionary, corpus, texts, limit, start, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("nb topics : " + str(num_topics))
        model = ldamallet.LdaMallet(mallet_path=path_to_mallet_binary, corpus=corpus, id2word=dictionary, num_topics=num_topics, random_seed =1, iterations = 200, alpha = 0.5)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

### Obtimal nb of topics chosen by maximisation of the coherence ###
def choose_nb_topics (dico, corpus, texts, start, limit , step=1, model = "Mallet") :
    if model == "Mallet" :
        model_list, coherence_values = compute_coherence_values_mallet(dictionary=dico, corpus=corpus, texts=texts, start=start, limit=limit, step=step)
    else :
        model_list, coherence_values = compute_coherence_values(dictionary=dico, corpus=corpus, texts=texts, start=start, limit=limit, step=step)
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    i = coherence_values.index(max(coherence_values))
    num_topics = i*step + start
    optimal_model = model_list[i]
    return(num_topics, optimal_model)

### LDA model creation, mallet gives better results ###
def create_lda(corpus,id2word, nb_topics, texts, v = -1, model = "Mallet", coherence = False) :
    if model == "Mallet" :
        lda = ldamallet.LdaMallet(mallet_path=path_to_mallet_binary, corpus=corpus, id2word=id2word, num_topics=nb_topics, random_seed =1, iterations = 500, alpha = 0.5)
        lda_model = ldamallet.malletmodel2ldamodel(lda)
    else :
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=nb_topics,
            random_state=100, update_every=1, chunksize=100, passes=1, iterations = 50, alpha='symmetric')
    
    print('\nTopics: \n',lda_model.print_topics())
    ### Perplexity measure, the smaller the better ###
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    ### Coherence measure, the higher the better ###
    if coherence :
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
    ### Visualisation ###
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    if v > -1 :
        lda_model.save('backup/models/test_numb_' + str(v)+"/"+str(v) + '.lda.model')
        pyLDAvis.save_html(vis, 'backup/models/test_numb_' + str(v)+"/"+str(v) +".visu.html")
        with open('backup/models/test_numb_' + str(v)+"/"+str(v) + ".corpus.txt", "wb") as fp:
            pickle.dump(corpus, fp)
    return (lda_model, vis)

def load_lda(v) :
    lda_model = gensim.models.ldamodel.LdaModel.load('backup/models/test_numb_' + str(v)+"/"+str(v) + ".lda.model")
    id2word = lda_model.id2word
    with open('backup/models/test_numb_' + str(v)+"/"+str(v) + ".corpus.txt", "rb") as fp:
        corpus = pickle.load(fp)
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    return(lda_model, id2word, corpus,vis)

######################################## Topic Modeling ########################################









import numpy as np
import networkx as nx
from pyvis.network import Network

def create_topic_topic_mat(lda_model):
    topic_term_mat = lda_model.get_topics()
    nb_topics = len(topic_term_mat)
    topic_topic_mat = np.zeros((nb_topics,nb_topics))
    for ix,iy in np.ndindex(topic_topic_mat.shape):
        topic_topic_mat[ix,iy] = np.dot(topic_term_mat[ix,:],topic_term_mat[iy,:])
    return topic_topic_mat

def visu_graph (topic_topic_mat, info):
    nb_topics = len (topic_topic_mat)
    net = Network()
    net.add_nodes(info.Cluster_nb, value = info.Perc_size, label = info.Label)
    t = np.mean(topic_topic_mat)
    for ix in range(nb_topics):
        for iy in range(ix):
            if topic_topic_mat[ix,iy] > t :
                net.add_edge(ix, iy, weight= topic_topic_mat[ix,iy], value = topic_topic_mat[ix,iy])
    #plt.figure()
    #nx.draw_shell(G, with_labels=True, font_weight='bold')
    #plt.show()
    #net.enable_physics(True)
    net.show_buttons(filter_=['physics'])
    net.show('mygraph.html')

def colorize (text, token2id, lda_model):
    L = []
    for x in text :
        try :
            l = lda_model.get_term_topics(token2id[x], minimum_probability = 0)
            l = sorted(l, key=lambda x: (x[1]), reverse=True)
            L.append((x,l[0][0]))
        except :
            pass
    return L




