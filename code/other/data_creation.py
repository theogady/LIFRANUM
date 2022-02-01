#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:21:36 2022

@author: theogady
"""

import pandas as pd

from os import listdir
from os.path import isfile

import json

### DataFrame creation from blogger_blogs file, created from  Blogger API ###
def creation_data(blogger_file):
    dossier = blogger_file
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
    data.to_pickle("backup/data/raw_data")