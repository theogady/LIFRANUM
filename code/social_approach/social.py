#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:22:55 2021

@author: theogady
"""
from bs4 import BeautifulSoup
import requests
import re

import tqdm

import pandas as pd


######################################## Analyse réseaux social ########################################

def get_followers(url):
#plutot que de changer les erreurs d'encoding à la main, ne pas ouvrir avec le bon encodeur directement ?
    try :
        r = requests.get(url)
    except :
        print("Impossible de se connecter à : "+ url)
        return False
    soup = BeautifulSoup(r.text, 'html.parser')
    followers = soup.find("div", id='Followers1')
    if followers is None :
        print("Pas de followers disponibles : "+ url)
        return False
    followers = re.search('followersIframeOpen\(".*', str(followers))
    followers = re.search('".*?"', followers.group(0)).group(0)[1:-1]
    followers = re.sub("\\\\x3d", '=', followers)
    followers = re.sub("\\\\x26", '&', followers)
    followers = re.sub("&pageSize=.*&origin", "&pageSize=1000&origin", followers)
    
    r = requests.get(followers)
    soup = BeautifulSoup(r.text, 'html.parser')
    members = soup.find_all("div", class_="member-thumbnail")
    df = pd.DataFrame(columns = ["name", "url","list_blogs","image"])
    for i in tqdm.tqdm(members) :
        link = i.a.get('href')
        link = re.search('".*?"', link).group(0)[1:-1]
        link = re.sub("\\\\x3d", '=', link)
        link = re.sub("\\\\x26", '&', link)
        
        r = requests.get(link)
        soup = BeautifulSoup(r.text, 'html.parser')
        content = soup.find("div", class_ ="content-container")
        
        display = content.find("div", class_ ="display-picture")
        display = display.find("img")
        name = display.get("alt")
        image = display.get("src")

        list_blogs = get_blogs(link, 0)
        
        df = df.append({"name": name, "url":link,"list_blogs":list_blogs,"image":image}, ignore_index=True)
        
    return(df)

def get_blogs(url, i):
    if i == 10 :
        return []
    print("Page numero " + str(i))
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    content = soup.find("div", class_ ="content-container")
    lim = 0
    while content is None and lim < 10 :
        print("-----je reessaie-----")
        lim += 1
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        content = soup.find("div", class_ ="content-container")
    if lim == 10 :
        print("-----impossible de lire la page-----")
        return []
    next_page = content.find( class_ = "link-active", id = "next")
    list_blogs = content.find_all( "a", class_ = False)
    list_blogs = [x.get("href") for x in list_blogs]
    if (not next_page is None) :
        new_url = "https://www.blogger.com" + next_page.get("href")[1:]
        list_blogs = list_blogs + get_blogs(new_url, i + 1)
    return list_blogs


def create_followers_data(list_urls, save):
    # 40 blogs avec abonnés dispo
    a = list_urls
    l = []
    for x in a :
        print("blog nb " + str(len(l)) + " / " + str(len(a)))
        l.append(get_followers(x))
    
    df = pd.DataFrame(columns = ["blog", "name", "url","list_blogs","image"])
    for i in range(len(l)):
        if isinstance(l[i], (bool)) :
            pass
        else :
            d = l[i]
            d["blog"] = a[i]
            df = df.append(l[i])
    notify("Fin programme")
    df.to_pickle("Svg_data/" + save)
    return df

import os 

def notify(text, title = "alert", sound = "default"):
    os.system("""osascript -e 'display notification "{}" with title "{}" sound name "{}" '""".format(text, title, sound))


from numpy import random
from time import sleep
import socket
def create_followers_data2(data):
    a = [x for x in set(data["author.url"]) if x == x and "blogger.com/profile/" in x]
    df = pd.DataFrame(columns = ["user_url", "My_blogs", "Blogs_followed","Info"])
    for x in tqdm.tqdm(a) :
        l1 = []
        l2 = []
        info = dict()
        #sleeptime = random.uniform(2,4)
        #sleep(sleeptime)
        r = requests.get(x)
        soup = BeautifulSoup(r.text, 'html.parser')
        i = 0
        while "Blogger: " not in soup.title.text :
            i+=1
            notify("Changement IP")
            sleeptime = 10
            print("try nb:", i)
            print(soup.title.text)
            print("sleeping for:", sleeptime, "seconds")
            sleep(sleeptime)
            print("sleeping is over")
            r = requests.get(x, stream = True)
            soup = BeautifulSoup(r.text, 'html.parser')
        content = soup.find("div", class_ ="contents-after-sidebar")
        if content is not None :
            for item in content.find_all("li", class_ ="sidebar-item"):
                if item.find("span") is None :
                    l2.append(item.a["href"])
                else :
                    l1.append(item.a["href"])
        table = soup.table
        if table is not None :
            for item in table.find_all("tr"):
                info[item.th.text] = item.td.text
        df = df.append({"user_url" : x, "My_blogs" : l1, "Blogs_followed" : l2, "Info" : info}, ignore_index=True)
    df.to_pickle("Svg_data/Followers2")
    return df


def create_followers_data2_bis(data):
    a = [x for x in set(data["author.url"]) if x == x and "blogger.com/profile/" in x]
    dico = dict()
    for user in a :
        url = data.loc[data["author.url"] == user]["blog.url"].values[0]
        if url in dico.keys() :
            dico[url].append(user)
        else :
            dico[url] = [user]
    df = pd.DataFrame(columns = ["user_url", "My_blogs", "Blogs_followed","Info"])
    count = 0
    for blog in tqdm.tqdm(dico.keys()) :
        for user in tqdm.tqdm(dico[blog]) :
            print("user nb:", count)
            count +=1
            l1 = []
            l2 = []
            info = dict()
            #sleeptime = random.uniform(2,4)
            #sleep(sleeptime)
            r = requests.get(blog)
            r = requests.get(user)
            soup = BeautifulSoup(r.text, 'html.parser')
            i = 0
            while "Profil" not in soup.title.text :
                i+=1
                sleeptime = random.uniform(120, 360)
                print("try nb:", i)
                print(soup.title.text)
                print("sleeping for:", sleeptime, "seconds")
                sleep(sleeptime)
                print("sleeping is over")
                soup = BeautifulSoup(r.text, 'html.parser')
            content = soup.find("div", class_ ="contents-after-sidebar")
            if content is not None :
                for item in content.find_all("li", class_ ="sidebar-item"):
                    if item.find("span") is None :
                        l2.append(item.a["href"])
                    else :
                        l1.append(item.a["href"])
            table = soup.table
            if table is not None :
                for item in table.find_all("tr"):
                    info[item.th.text] = item.td.text
            df = df.append({"user_url" : user, "My_blogs" : l1, "Blogs_followed" : l2, "Info" : info}, ignore_index=True)
    df.to_pickle("Svg_data/Followers2")
    return df

import itertools
from pyvis.network import Network
from igraph import Graph
import matplotlib.pyplot as plt
import igraph as ig
def create_graph(data, raw_data, save = False, show = False):
    nodes = set(data.blog.values)
    nodes.update([blog for l in data.list_blogs.values for blog in l])
    nodes = list(nodes)
    g = Graph()
    g.add_vertices(len(nodes))
    g.vs["blog.url"] = nodes
    list_raw = set(raw_data["blog.url"].values)
    g.vs["in_raw_data"] = [x in list_raw for x in nodes]
    dico_iD = dict(zip(nodes, range(len(nodes))))
    edges = dict()
    for i,row in tqdm.tqdm(data.iterrows()):
        l = list(set(row.list_blogs))
        origin = row.blog
        if origin == origin and origin not in l :
            l.append(origin)
        pairs = itertools.combinations(l, 2)
        for p in pairs :
            if p in edges.keys():
                edges[p] += 1
            elif (p[1],p[0]) in edges.keys() :
                edges[(p[1],p[0])] += 1
            else :
                edges[p] = 1
    l_items = list(edges.items())
    l_edges = [(dico_iD[x],dico_iD[y]) for (x,y), z in l_items if z > 1]
    l_values = [z for (x,y), z in l_items if z > 1]
    g.add_edges(l_edges)
    g.es["value"] = l_values
    g.vs["label"] = g.vs["blog.url"]
    g.vs["color"] = ["red" if x else "blue"for x in g.vs["in_raw_data"]]
    if show == "small" :
        nx = g.to_networkx()
        nt = Network()
        nt.from_nx(nx)
        nt.show_buttons(filter_=['physics'])
        nt.show('social_graph.html')
    if show == "big" :
        fig, ax = plt.subplots()
        layout = g.layout("drl")
        ig.plot(g,layout=layout, target=ax,vertex_size=2)
    if save :
        g.save('graph.pkl', format = 'pickle')
    return g

def histogram(values):
    hist = {}
    for v in values:
        if v in hist.keys():
            hist[v] += 1
        else:
            hist[v] = 1
    return hist


from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
import numpy as np
from distinctipy import distinctipy


def graph_clustering(graph, n_embedding, n_clusters):
    g = graph.copy()
    data_cluster = pd.DataFrame(columns = ["Cluster", "Value"])
    colors = ["rgb" + str((256*x[0],256*x[1],256*x[2])) + "" for x in distinctipy.get_colors(n_clusters)]
    color_dict = dict(zip(range(n_clusters),colors))
    X = g.get_adjacency_sparse()
    embedding = SpectralEmbedding(n_components=n_embedding, affinity = "precomputed", random_state = 1)
    X_transformed = embedding.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init = 100)
    clusters_values = kmeans.fit_transform(X_transformed)
    for row in clusters_values :
        cluster = np.argmin(row)
        perc = row[cluster]
        data_cluster = data_cluster.append({"Cluster" : cluster, "Value" : perc}, ignore_index=True)
    g.vs["cluster"] = data_cluster.Cluster
    g.vs["color"] = data_cluster.Cluster.apply(color_dict.get)
    g.vs["size"] = data_cluster.Value
    ig.plot(g, layout=graph.layout("drl"),vertex_size=5)
    return g

def graph_clustering2(graph, n_clusters = None):
    g = graph.copy()
    dendogram = g.community_edge_betweenness(n_clusters, weights = "value")
    
    if n_clusters == None :
        n_clusters = dendogram.optimal_count
        print(str(n_clusters) + " : clusters detected")
    clusters = dendogram.as_clustering()
    clusters = g.community_multilevel(weights = "value")

    colors = ["rgb" + str((256*x[0],256*x[1],256*x[2])) + "" for x in distinctipy.get_colors(len(clusters))]
    color_dict = dict(zip(range(len(clusters)),colors))
    g.vs["cluster"] = clusters.membership
    g.vs["color"] = [color_dict[i] for i in clusters.membership]
    return g
    
def addInfo(data, profiles):
    
    
    
    
    
    