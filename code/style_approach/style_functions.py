#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:31:21 2021

@author: theogady
"""
# Adapted from code of Enzo TERREAU
# see https://github.com/EnzoFleur/style_embedding_evaluation.git

import nltk

from nltk import word_tokenize

from collections import Counter

import math



import numpy as np

cmuDictionary = nltk.corpus.cmudict.dict()

###
from config import french_stopwords, nlp

### Lexical Features ###

# Lexical feature (word level)
# Average word length of a document
def average_word_length(words):
    if len(words)==0:
        return 0
    average = sum(len(word) for word in words)/(len(words))
    return average
### Total number of short words (length <4) in a document ###
def total_short_words(words):
    count_short_word = 0
    if len(words)==0:
        return 0
    for word in words:
        if len(word) < 4:
            count_short_word += 1
    return count_short_word/(len(words))
### Average number of digit in document ###
def total_digit(text_doc):
    return sum(c.isdigit() for c in text_doc)/(len(text_doc))
### Average number of uppercase letters in document ###
def total_uppercase(text_doc):
    return sum(1 for c in text_doc if c.isupper())/(len(text_doc))
### Letter frequency in document ###
def count_letter_freq(text_doc):
    text_doc = ''.join([i.lower() for i in text_doc if i.isalpha()])
    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's'
              , 't', 'u', 'v', 'w', 'x', 'y', 'z']
    count = {}
    text_length=len(text_doc)
    for s in text_doc:
      if s in count.keys():
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in letter:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0
    return({ll:count[ll]/text_length if ll in count.keys() else 0 for ll in letter})
### Digit frequency in document ###
def count_digit_freq(text_doc): 
    text_doc = ''.join([i for i in text_doc if i.isdigit()])
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    count = {}
    text_length = len(text_doc)
    for s in text_doc:
      if s in count.keys():
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in digits:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0

    return({dd:count[dd]/text_length if dd in count.keys() else 0 for dd in digits})
### Average sentence length in a document ###
def average_sentence_length(sent_list):
    average = sum(len(nltk.word_tokenize(sent)) for sent in sent_list)/(len(sent_list))
    return average
### Lexical Feature (vocabulary richness) ###
def hapax_legomena_ratio(words):  # # per document only a float value
    fdist = nltk.FreqDist(word for word in words)
    fdist_hapax = nltk.FreqDist.hapaxes(fdist)
    return float(len(fdist_hapax)/(len(words)))
def dislegomena_ratio(words):  # per document only a float value
    vocabulary_size = len(set(words))
    freqs = Counter(nltk.probability.FreqDist(words).values())
    VN = lambda i:freqs[i]
    return float(VN(2)*1./(vocabulary_size))
def CountFunctionalWords(words):

    count = 0

    for i in words:
        if i in french_stopwords:
            count += 1

    return count / len(words)
def freq_function_word(words):  # per document (vector with length ?)
    count = {}
    n_words=len(words)
    for s in words:
      if s in count.keys():
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in french_stopwords:
        if d in count.keys():
            count_list[d] = count[d]/n_words
        else:
            count_list[d] = 0
    return {ww:(count_list[ww]) for ww in french_stopwords}
def punctuation_freq(text):
    punct = ['\'', ':', ',', '_', '!', '?', ';', ".", '\"', '(', ')', '-', "#", "@", "/"]
    count = {}
    text_length=len(text)
    for s in text:
      if s in count.keys():
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in punct:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0

    return({pp:count[pp]/text_length if pp in count.keys() else 0 for pp in punct})
def syllable_count_Manual(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
### cmuDictionary que en anglais ? ###
def syllable_count(word):
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in cmuDictionary[word.lower()]][0]
    except:
        syl = syllable_count_Manual(word)
    return syl
def Avg_Syllable_per_Word(words):
    syllabls = [syllable_count(word) for word in words]
    return sum(syllabls) / max(1, len(words))
# Problème ds code Antoine ? car word_tokenise ne sépare pas les apostrophes
def RemoveSpecialCHs(text):
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?","@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    for char in text:
        if char in st:
            text = text.replace(char,' ')
    words = word_tokenize(text)
    return words
def AvgWordFrequencyClass(words):
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])
# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
# grand varié
def YulesCharacteristicK(words):
    N = len(words)
    freqs = Counter()
    freqs.update(words)
    vi = Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K
# -1*sigma(pi*lnpi)
# Shannon and sympsons index are basically diversity indices for any community
## grand == varié
def ShannonEntropy(words):
    lenght = len(words)
    freqs = Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)
    import scipy as sc
    H = sc.stats.entropy(distribution, base=2)
    return H
# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
## petit == vocab varié
def SimpsonsIndex(words):
    freqs = Counter()
    freqs.update(words)
    N = len(words)
    if N<=1:
        return 0
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    return - D
## petit == difficile à lire
def FleschReadingEase(words, NoOfsentences):
    l = float(len(words))
    scount = 0
    for word in words:
        scount += syllable_count(word)

    I = 206.835 - 1.015 * (l / float(NoOfsentences)) - 84.6 * (scount / float(l))
    return - I
## donne niveau grade
def FleschCincadeGradeLevel(words, NoOfSentences):
    scount = 0
    for word in words:
        scount += syllable_count(word)

    l = len(words)
    F = 0.39 * (l / NoOfSentences) + 11.8 * (scount / float(l)) - 15.59
    return F
## grand == complex
def dale_chall_readability_formula(words, NoOfSectences):
    difficult = 0
    adjusted = 0
    NoOfWords = len(words)

    for word in words:
        if word not in french_stopwords:
            difficult += 1
    percent = (difficult / NoOfWords) * 100
    if (percent > 5):
        adjusted = 3.6365
    D = 0.1579 * (percent) + 0.0496 * (NoOfWords / NoOfSectences) + adjusted
    return D
## grand == complex
def GunningFoxIndex(words, NoOfSentences):
    NoOFWords = float(len(words))
    complexWords = 0
    for word in words:
        if (syllable_count(word) > 2):
            complexWords += 1

    G = 0.4 * ((NoOFWords / NoOfSentences) + 100 * (complexWords / NoOFWords))
    return G

def pos_freq(text, NoOfSentences):      
    doc = nlp(text)
    features_dict={
    **{k:v/NoOfSentences for k, v in Counter([token.tag_ for token in doc]).items()},
    **{k:v/NoOfSentences for k,v in Counter([entity.label_ for entity in doc.ents]).items()}
        }
    return(features_dict)

def create_feature(text, words, sent_text):
    stylometry = {}
    # A quoi ça sert ?
    #text = ''.join(sent_text)
    NoOfSentences = len(sent_text)
    stylometry['avg_w_len']=average_word_length(words)
    stylometry['tot_short_w']=total_short_words(words)
    stylometry['tot_digit']=total_digit(text)
    stylometry['tot_upper']=total_uppercase(text)
    stylometry={**stylometry, **count_letter_freq(text)}
    stylometry={**stylometry, **count_digit_freq(text)}
    stylometry['avg_s_len']=average_sentence_length(sent_text)
    stylometry['hapax']=hapax_legomena_ratio(words)
    stylometry['dis']=dislegomena_ratio(words)
    stylometry['func_w_freq']=CountFunctionalWords(words)
    stylometry={**stylometry, **freq_function_word(words)}
    stylometry={**stylometry,**punctuation_freq(text)}
    stylometry["syllable_count"]=Avg_Syllable_per_Word(words)
    stylometry["avg_w_freqc"]=AvgWordFrequencyClass(words)
    stylometry['yules_K']=YulesCharacteristicK(words)
    stylometry['shannon_entr']=ShannonEntropy(words)
    stylometry['simposons_ind']=SimpsonsIndex(words)
    stylometry['flesh_ease']=FleschReadingEase(words, NoOfSentences)
    stylometry['flesh_cincade']=FleschCincadeGradeLevel(words, NoOfSentences)
    stylometry['dale_call']=dale_chall_readability_formula(words, NoOfSentences)
    stylometry['gunnin_fox']=GunningFoxIndex(words, NoOfSentences)
    stylometry={**stylometry, **pos_freq(text, NoOfSentences)}   
    return stylometry