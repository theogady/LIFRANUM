#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:25:50 2022

@author: theogady
"""
from fonction_import.Topic_Model_functions import load_lda
from igraph import Graph
import pandas as pd
import numpy as np
from fonction_import.fr_extractor import create_feature, sent_tokenize, RemoveSpecialCHs, clean_str
import tqdm
from fonction_import.fonctions_utiles import html_to_text, clean_text, getlang
from gensim.models.doc2vec import Doc2Vec
from config import french_stopwords
from sklearn.metrics.pairwise import cosine_similarity as cosim
from heapq import nlargest
from fonction_import.blogger_collector import get_blog_info, get_blog_posts
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

filtre_stopfr =  lambda text: [token for token in text if token.lower() not in french_stopwords]

# def Topics_score(text, lda_model, topics_values, min_proba = 0.2):
    id2word = lda_model.id2word
    new_text = id2word.doc2bow(text)
    topics = lda_model.get_document_topics(new_text, min_proba)
    num = 0
    denom = 0
    for topic, perc in topics :
        tv = topics_values[topic]
        if tv > 0 :
            num += tv *perc
            denom += tv
        else :
            num += - tv *(1-perc)
            denom += -tv
    if denom == 0 :
        return 0
    else :
        return num/denom

def Topics_score(text, lda_model, topics_values):
    id2word = lda_model.id2word
    new_text = id2word.doc2bow(text)
    topics = lda_model.get_document_topics(new_text, 0.0)
    l = [topics_values[topic]*perc for topic, perc in topics]
    return sum(l)

def get_style(text, max_sentence=300):
    sent_text = sent_tokenize(clean_str(text), space=True)[:max_sentence]
    
    text = ''.join(sent_text)
    words=RemoveSpecialCHs(text)
    
    return create_feature(text, words, sent_text)

# def Style_score(text, words, sent_text, style_coordinates, points_values, k = 10):
    style_dict = create_feature(text, words, sent_text)
    commun_features = set(style_dict.keys()).intersection(set(style_coordinates.columns))
    style_coordinates_adapted = style_coordinates[commun_features]
    new_point = [style_dict[c]  for c in style_coordinates_adapted.columns]
    X = np.array(new_point).reshape(1,-1)
    
    scaler = StandardScaler()
    scaler.fit(style_coordinates_adapted)
    style_coordinates_scaled = scaler.transform(style_coordinates_adapted)
    X = scaler.transform(X)
    
    sims = cosim(X,style_coordinates_scaled)[0]
    sims = list(zip(sims, range(len(sims))))
    sims = nlargest(k,sims)
    return sum([points_values[id_doc]*sim for sim, id_doc in sims])/k

# def Style_score(text, words, sent_text, style_coordinates, points_values, k = 10):
    style_dict = create_feature(text, words, sent_text)
    commun_features = set(style_dict.keys()).intersection(set(style_coordinates.columns))
    style_coordinates_adapted = style_coordinates[commun_features]
    new_point = [style_dict[c]  for c in style_coordinates_adapted.columns]
    X = np.array(new_point).reshape(1,-1)
    
    scaler = StandardScaler()
    scaler.fit(style_coordinates_adapted)
    style_coordinates_scaled = scaler.transform(style_coordinates_adapted)
    X = scaler.transform(X)
    
    sims = cosim(X,style_coordinates_scaled)[0]
    sims = list(zip(sims, range(len(sims))))
    sims = nlargest(k,sims)
    
    num = 0
    denom = 0
    for id_doc,sim in sims :
        pv = points_values[id_doc]
        if pv > 0 :
            num += pv*sim
            denom += pv
        else :
            num += -pv*(1-sim)
            denom += - pv
    return num/denom

def Style_score2(text, words, sent_text, style_coordinates):
    style_dict = create_feature(text, words, sent_text)
    commun_features = set(style_dict.keys()).intersection(set(style_coordinates.columns))
    style_coordinates_adapted = style_coordinates[commun_features]
    new_point = [style_dict[c]  for c in style_coordinates_adapted.columns]
    X = np.array(new_point).reshape(1,-1)
    
    scaler = StandardScaler()
    style_coordinates_scaled = scaler.fit_transform(style_coordinates_adapted)
    X = scaler.transform(X)
    
    transform = Nystroem(kernel = "rbf", random_state=1)
    clf_sgd = SGDOneClassSVM(nu = 10e-3, random_state=1)
    pipe_sgd = make_pipeline(transform, clf_sgd)
    # clf = OneClassSVM(gamma='auto').fit(style_coordinates_scaled)
    # lof = LocalOutlierFactor(novelty=True).fit(style_coordinates_scaled)
    
    pipe_sgd.fit(style_coordinates_scaled)
    return(pipe_sgd.decision_function(X)[0])


# def point_evaluation(new_point, old_points, points_values, radius = 1):
#     distances = np.sum((old_points - new_point)**2, axis = 1)
#     l = [1/d*points_values[i] for (i,d) in enumerate(distances) if d <= radius]
#     return np.sum(l)

# def Content_score(text, embedding_model, content_coordinates, points_values, k = 10):
    new_point = embedding_model.infer_vector(text)
    sims = embedding_model.dv.most_similar([new_point], topn=k)
    sims = [(id_doc,(1 + sim)/2) for id_doc,sim in sims]
    num = 0
    denom = 0
    for id_doc,sim in sims :
        pv = points_values[id_doc]
        if pv > 0 :
            num += pv*sim
            denom += pv
        else :
            num += -pv*(1-sim)
            denom += - pv
    return num/denom

def Content_score2(text, embedding_model, content_coordinates):
    new_point = embedding_model.infer_vector(text)
    transform = Nystroem(kernel = "rbf", random_state=1)
    clf_sgd = SGDOneClassSVM(nu = 10e-3, random_state=1)
    pipe_sgd = make_pipeline(transform, clf_sgd)
    pipe_sgd.fit(content_coordinates)
    X = np.array(new_point).reshape(1,-1)
    return(pipe_sgd.decision_function(X)[0])

def Main(lda_model, topics_values, social_graph, style_coordinates, embedding_model, content_coordinates, points_values):
    
    return 0

lda_model , _, _,_ = load_lda(14)
topics_values = dict(zip(range(30), [1]*30))
social_graph = Graph.Read_Pickle("Svg_graph/AandF.pkl")
# à centrer reduire
style_coordinates = pd.read_csv("/Users/theogady/Stage_eric/blogger_posts_style.csv", sep = ";").drop(["author_id","author","doc_id"], axis = 1)
scaler = StandardScaler()
scaler.fit(style_coordinates)
style_coordinates_scaled = scaler.transform(style_coordinates)

embedding_model = Doc2Vec.load("Svg_docEmbedding/Model_1")
content_coordinates = embedding_model.dv.get_normed_vectors()
points_values = [1]*109017

###
Neut = [3,8,2,24]
Neg = [1, 10, 29]
for i in Neut :
    topics_values[i] = 0
for i in Neg :
    topics_values[i] = -1
### Données pour tester ###
raw_data = pd.read_pickle("Svg_data/Data")
html = raw_data.iloc[0].content
text = html_to_text(html)

words2 = clean_text(text)

max_sentence = 300
sent_text = sent_tokenize(clean_str(text), space=True)[:max_sentence]

text = ''.join(sent_text)
words=RemoveSpecialCHs(text)


cs = Content_score2(words2, embedding_model, content_coordinates)
ss = Style_score2(text, words, sent_text, style_coordinates)
ts = Topics_score(words, lda_model, topics_values)

### online ###


scores = []

for blog_url in tqdm.tqdm(social_graph.vs.select(lambda vertex: not vertex["in_raw_data"])["blog.url"]) :
    print("\n")
    print(blog_url)
    try :
        blog_id = get_blog_info(blog_url)['id']
        posts = get_blog_posts(blog_id, max_pages= 2, posts_per_page=1)
    except :
        posts = []

    for post in posts :
        content = post["content"]
        text = html_to_text(content)
        words2 = clean_text(text)
        lang = getlang(text)
        max_sentence = 300
        sent_text = sent_tokenize(clean_str(text), space=True)[:max_sentence]

        text = ''.join(sent_text)
        words=RemoveSpecialCHs(text)
        if len(words) == 0 or getlang(words) != "FRENCH":
            scores.append((0,(0,0,0),post["url"]))
        else :
            #besoin meme format que apprentissage 
            
            cs = Content_score2(words2, embedding_model, content_coordinates)
            ss = Style_score2(text, words, sent_text, style_coordinates)
            ts = Topics_score(words2, lda_model, topics_values)
            s = abs(cs * ss * ts)
            if cs < 0 or ss < 0 :
                s = -s
            print("\n {} : \n cs = {:.4f}, ss = {:.4f}, ts = {:.4f}".format(post['url'],cs,ss,ts))
            scores.append((s,(cs,ss,ts),post["url"]))
    if len(scores)%50 == 0:
        print(f"############### nb posts = {len(scores)} STOP ? ###############")
        time.sleep(5)
        
        


results = sorted(scores, reverse = True)

# r = [t for t in results if len(t) == 3]

r1 = sorted([(cs,p) for s,(cs,ss,ts),p in results], reverse = True)
r2 = sorted([(ss,p) for s,(cs,ss,ts),p in results], reverse = True)
r3 = sorted([(ts,p) for s,(cs,ss,ts),p in results], reverse = True)
# style + thematiques, pas mal, mais mauvais classement pour poésie
# tester embedding sur phrases completes OU avec camemBert OU revenir à la mesure sans SVM
r4 = sorted([(ts*ss,p) for s,(cs,ss,ts),p in results], reverse = True)

#scipy distance matrice
# one class svm
