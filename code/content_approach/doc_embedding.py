#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:56:49 2022

@author: theogady
"""
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

n = 2

cleaned_data = pd.read_pickle("Svg_data/Cleaned_Data")
data_posts = cleaned_data[(cleaned_data.language == "FRENCH") & (cleaned_data.kind == "blogger#post")]

texts = data_posts.content.values

readable_data = pd.read_pickle("Svg_data/Readable_Data")
texts = readable_data.content.values
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
model = Doc2Vec(documents, vector_size=50, min_count=2, epochs=40)

model.save("Svg_docEmbedding/Model_" + str(n))

coor = model.dv.get_normed_vectors()

import tqdm
train_corpus = list(documents)

ranks = []
second_ranks = []
for doc_id in tqdm.tqdm(range(len(train_corpus))):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

import collections

counter = collections.Counter(ranks)
print(counter)

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

import transformers
from transformers.tokenization_camembert import CamembertTokenizer
TOKENIZER = CamembertTokenizer.from_pretrained('camembert')
import torch
camembert = torch.hub.load('pytorch/fairseq', 'camembert')
sentence = "j'aime le chocolat"
r = TOKENIZER.tokenize(sentence)
data = texts.tolist()



