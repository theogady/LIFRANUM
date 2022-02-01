#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:36:54 2021

@author: theogady
"""
######################################## Packages ########################################

import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

import pandas as pd

import matplotlib.pyplot as plt

import tqdm

import random
random.seed(1)


######################################## Topics clusterisation ########################################

### Clusterisation of the posts regarding their main topic, some irrelevant topics will not be taken into account ###
def naive_topics_clustering (data, ldamodel, v = -1, irr_topics = []):
    copy = data
    classes = []
    perct = []
    ### Creation of the term/document matrix ###
    corpus = []
    id2word = ldamodel.id2word
    for text in data.content.values.tolist() :
        corpus.append(id2word.doc2bow(text))
    ### Get main topic in each document ###
    for i, row in tqdm.tqdm(enumerate(ldamodel[corpus])):
        ### Changes between mallet or gensim implementation
        if type(row) == tuple :
            row = row[0]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        j = 0
        while j < len(row):
            topic = row[j][0]
            if topic in irr_topics :
                j += 1
            else :
                classes.append(topic)
                perct.append(row[j][1])
                break
        if j == len(row):
            classes.append(row[0][0])
            perct.append(row[0][1])
    copy["cluster"] = classes
    copy["proba"] = perct
    if v > -1 :
        copy.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v)+".classified_posts")
    return(copy)

### Clusterisation in topics space with k-means ###
def topic_vector_clustering(data, ldamodel, n_clusters, v = -1):
    copy = data.copy()
    ### Creation of the term/document matrix ###
    corpus = []
    id2word = ldamodel.id2word
    for text in data.content.values.tolist() :
        corpus.append(id2word.doc2bow(text))
    classes = []
    perct = []
    X = []
    n_topics = len(ldamodel.print_topics(num_topics=-1))
    ### WARNING ldamodel[corpus]  is non deterministic ###
    for i, row in tqdm.tqdm(enumerate(ldamodel[corpus])):
        ### Changes between mallet or gensim implementation
        if type(row) == tuple :
            row = row[0]
        topics_coord = [0]*n_topics
        for x in row :
            topics_coord[x[0]] = x[1]
        X.append(topics_coord)
    kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init = 100)
    coord = kmeans.fit_transform(X)
    for row in coord :
        cluster = np.argmin(row)
        perc = row[cluster]
        classes.append(cluster)
        perct.append(perc)
    copy["cluster"] = classes
    copy["proba"] = perct
    if v > -1 :
        copy.to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v)+".classified_posts")
        pd.DataFrame(kmeans.cluster_centers_).to_pickle('backup/topic_models/test_numb_' + str(v)+"/"+str(v)+".centroids")
        pd.DataFrame(X).to_pickle('backup/models/test_numb_' + str(v)+"/"+str(v)+".coord")
    return(copy, kmeans.cluster_centers_, X)

def style_classifier(data, X, n_clusters):
    copy = data.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init = 100)
    classes = []
    perct = []
    coord = kmeans.fit_transform(X)
    for row in coord :
        cluster = np.argmin(row)
        perc = row[cluster]
        classes.append(cluster)
        perct.append(perc)
    copy["Class"] = classes
    copy["Proba"] = perct
    return(copy, kmeans.cluster_centers_, X)

def visu_clusters(data,X, centers, L_labels, n_sample = 3000, focus = []) :
    classes = list(data.Class.values)
    n_clusters = len(set(classes))
    embedding = TSNE(n_components=2, random_state = 0)
    random.seed(1)
    X_sample = random.sample(X, n_sample)
    random.seed(1)
    classes_sample = random.sample(classes, n_sample)
    focus_sample = []
    for column, value in focus :
        l =list(data[column].values)
        random.seed(1)
        focus_sample.append(random.sample(l, n_sample))
    X_sample =  X_sample + list(centers)
    X_2D = embedding.fit_transform( X_sample)
    centroids_2D = X_2D[-n_clusters:]
    X_2D = X_2D[:-n_clusters]
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i) for i in np.linspace(0.1, 1, n_clusters)]
    if len(focus) == 0 :
        for g in np.unique(classes_sample):
            ix = np.where(classes_sample == g)
            ax.scatter(X_2D[ix,0], X_2D[ix,1], c= colors[g])
    else :
        #cmap = plt.get_cmap('copper')
        colors = [cmap(i) for i in np.linspace(0.1, 1, len(focus))]
        for f in range(len(focus)):
            ix = [ idx for idx, x in enumerate(focus_sample[f]) if x == focus[f][1]]
            ax.scatter(X_2D[ix,0], X_2D[ix,1], c= colors[f], label = str(focus[f]), marker = "d")
            ax.legend()
    for g in np.unique(classes_sample):
        #ax.annotate(g, (centroids_2D[g,0],centroids_2D[g,1]))
        ax.annotate(L_labels[g], (centroids_2D[g,0],centroids_2D[g,1]), fontsize=10, fontweight = 'bold')
        ax.scatter(centroids_2D[g,0],centroids_2D[g,1],marker = "x", c ="black")
    plt.show()
    
    