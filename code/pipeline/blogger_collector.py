#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:28:14 2022

@author: theogady
"""
# Adapted from code of Javier ESPINOSA
# see https://github.com/javieraespinosa/lifranum.git

BLOG_URL  = 'https://poesiecls.blogspot.com/'

MY_APPLICATION_KEY = ""

#from google.colab import drive
#drive.mount("/content/gdrive")

import requests 
import json
import time

# Gets blog's general information based on its URL
def get_blog_info(blog_url): 
    enpoint="https://www.googleapis.com/blogger/v3/blogs/byurl"
    params = {
        'url': blog_url,
        'key': MY_APPLICATION_KEY
    } 
    r = requests.get(url=enpoint, params=params)
    data = r.json() 
    return data

# Gets blog's posts given a blog's id. You can control the max number of pages to
# retrieve and the number of posts per page. By default, the function will collect
# all blog's posts.
def get_blog_posts(blog_id, max_pages=0, posts_per_page=50):
    posts = []
    data = None
    p = 1
    try:
        endpoint = "https://www.googleapis.com/blogger/v3/blogs/{}/posts".format(blog_id)
        params = {
            'key': MY_APPLICATION_KEY,
            'maxResults': posts_per_page
        }
        while True:
            r = requests.get(url=endpoint, params=params)
            data = r.json()
            posts.extend(data['items'])

            #print('last post:', data['items'][-1]['id'], data['items'][-1]['url'])

            if max_pages > 0 and p >= max_pages:
                break
            
            # Retrieve until there are no more pages left
            if 'nextPageToken' not in data:
                break

            params['pageToken'] = data['nextPageToken']

            # sleep every 2 calls to avoid google rate limits
            if p % 2 == 0:
                time.sleep(2)  
            p+=1

    except Exception as e :
        print('error:', e)
        print('data:', data)
  
    return posts

if __name__ == "__main__" :
    blog_info = get_blog_info(BLOG_URL)
    
    with open('blog_info.json', 'w') as file:
        json.dump(blog_info, file)
        
    blog_id = get_blog_info(BLOG_URL)['id']
    posts   = get_blog_posts(blog_id, max_pages=2)
    
    with open('blog_posts.json', 'w') as file:
        json.dump(posts, file)