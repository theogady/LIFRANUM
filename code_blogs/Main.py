#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:22:34 2021

@author: theogady
"""
######################################## Packages ########################################
import pandas as pd

from os import mkdir

###
from fonction_import.fonctions_utiles import creation_data, html_to_text, clean_text, getlang, limit_X, divide_data1, divide_data2, divide_data3, make_cloud, distrib_X, resize
from fonction_import.Topic_Model_functions import creation_corpus, create_lda, load_lda, create_lda_mallet, choose_nb_topics
from fonction_import.classifiers import naive_classifier, topic_vector_classifier
from fonction_import.analyse_cluster import ini_info_cluster, get_most_sallient_words, get_ngram,get_most_repr_text2,get_size_cluster, get_style, labelisation, visu_style, get_most_sallient_words2, get_ngram2, gene, hill_climbing, RandR
from fonction_import.analyse_gen import get_biggest_X, languages
from fonction_import.test import analyse_freq

######################################## Importation des données ########################################
#creation_data()

raw_data = pd.read_pickle("Svg_data/Data")
raw_data = raw_data.set_index("id")

raw_data_divided = pd.read_pickle("Svg_data/Divided_data")

# readable_data = raw_data.copy()
# readable_data["content"] = readable_data["content"].apply(html_to_text, args=[])
# readable_data.to_pickle("Svg_data/Readable_Data")
readable_data = pd.read_pickle("Svg_data/Readable_Data")


### Cleaning des données ###
# cleaned_data = readable_data.copy()
# cleaned_data["content"] = readable_data["content"].apply(clean_text, args=[])
# cleaned_data.to_pickle("Svg_data/Cleaned_Data")

### Evaluation langue ###
cleaned_data = pd.read_pickle("Svg_data/Cleaned_Data")
# cleaned_data["language"] = cleaned_data.content.apply(getlang, args = [])
# cleaned_data.to_pickle("Svg_data/Cleaned_Data")

data_posts = cleaned_data[(cleaned_data.language == "FRENCH") & (cleaned_data.kind == "blogger#post")]
data_posts_lim = limit_X(data_posts,500)

######################################## Infos des sauvegardes ########################################
#Infos_svg = pd.DataFrame(columns=["Description"])
#Infos_svg.to_pickle("Svg_analyse/Infos_svg")
Infos_svg = pd.read_pickle("Svg_analyse/Infos_svg")

v = 20
try:
   mkdir('Svg_analyse/Test_numb_' + str(v))
   print("Fichier créé")
except:
    print("Fichier existant")


######################################## Main ########################################
dico, corpus_lim, list_data, sample = creation_corpus(data_posts_lim, 1, total_text =True)
lda_model, vis = create_lda(corpus_lim, dico, 30, list_data, v = v)
# parfois erreur : returned non-zero exit status 1
lda_model2, vis2 = create_lda_mallet(corpus_lim, dico, 35, list_data, v = v)
# partie calcul coherence très long, enlevée, à deplacer


lda_model, id2word, corpus,vis = load_lda(14)

Infos_svg.loc[v] = ["Dico_1,Data_1,Topics_30, lim_dico min 5 max 0.5/corpus, max freq = 10/text, gestion espaces, longueur text, mallet ac 500 itérations, stop words de spacy, lim 500 posts/auteur, 40 clusters"]
Infos_svg.to_pickle("Svg_analyse/Infos_svg")
# pr corpus entier, à mettre comme option ds "création corpus"
corpus = []
for text in data_posts.content.values.tolist() :
    corpus.append(id2word.doc2bow(text))

classified = naive_classifier(data_posts,lda_model2,corpus, v)


classified, centroids, X = topic_vector_classifier(data_posts, lda_model, corpus, 40, v, save = True)
# pr affichage k-means, peut etre aussi faire un sample
ini_info_cluster(v)

get_most_sallient_words (v, Lambda = 0.5)

get_ngram (v, Lambda = 0.5)

get_most_repr_text2(v, nb_url = 10)

get_size_cluster(v)

L_labels =["corps","haikus+corps","festivals litt","quebecois oral","#2","nature","nature_bis","corps_bis","ana nb","petition_politique","langue orale","voyage ville photo", "WW2, russie, terrorisme", "bilan lecture", "orchestre","critique livre par DB","pkd", "violence humaine", "voyage mechanisé organisation", "questionnaire ecrivantes", "decription ouvrage litt", "poésie (danger)", "@1","famille","essai philo","verbes je/on","tournois nouvelles","nature + ana nb", "haiku","festivals litt_bis","verbes 3P passé", "ana nb","ecriture","petition politique + langue orale + essai philo","#1","corps + nature","theatre","famille","langue orale","poésie (danger) + corps + nature"]
labelisation(v, L_labels)

get_style(v,frac = 0.05)

visu_style(v, N_features = 8)


choose_nb_topics (dico, corpus_lim, list_data, 20, 40 , step=1, model = "Mallet")

info = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".info_clusters")
style = pd.read_pickle('Svg_analyse/Test_numb_' + str(v)+"/"+str(v) + ".style_eval")


get_biggest_X(raw_data,"author.id")

followers_data = pd.read_pickle("Svg_data/Followers")






df_style = pd.read_csv("/Users/theogady/Stage_eric/blogger_posts_style.csv", sep = ";")

