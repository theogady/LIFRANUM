#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:10:18 2021

@author: theogady
"""
# Adapted from code of Enzo TERREAU
# see https://github.com/EnzoFleur/style_embedding_evaluation.git

import pandas as pd
import numpy as np

import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from scipy import stats


SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

np.seterr(divide='ignore', invalid='ignore')

letters={'a': 'Letters',
    'b': 'Letters',
    'c': 'Letters',
    'd': 'Letters',
    'e': 'Letters',
    'f': 'Letters',
    'g': 'Letters',
    'h': 'Letters',
    'i': 'Letters',
    'j': 'Letters',
    'k': 'Letters',
    'l': 'Letters',
    'm': 'Letters',
    'n': 'Letters',
    'o': 'Letters',
    'p': 'Letters',
    'q': 'Letters',
    'r': 'Letters',
    's': 'Letters',
    't': 'Letters',
    'u': 'Letters',
    'v': 'Letters',
    'w': 'Letters',
    'x': 'Letters',
    'y': 'Letters',
    'z': 'Letters'}
numbers={'tot_digit': 'Numbers',
    '0': 'Numbers',
    '1': 'Numbers',
    '2': 'Numbers',
    '3': 'Numbers',
    '4': 'Numbers',
    '5': 'Numbers',
    '6': 'Numbers',
    '7': 'Numbers',
    '8': 'Numbers',
    '9': 'Numbers'}
ner={'LOC': 'NER',
    'MISC': 'NER',
    'ORG': 'NER',
    'PER': 'NER',}
tag={'ADJ': 'TAG',
    'ADP': 'TAG',
    'ADV': 'TAG',
    'AUX': 'TAG',
    'CCONJ': 'TAG',
    'DET': 'TAG',
    'INTJ': 'TAG',
    'NOUN': 'TAG',
    'NUM': 'TAG',
    'PART': 'TAG',
    'PRON': 'TAG',
    'PROPN': 'TAG',
    'PUNCT': 'TAG',
    'SCONJ': 'TAG',
    'SYM': 'TAG',
    'VERB': 'TAG',
    'X': 'TAG'}
function_words={'avaient': 'Function words',
 'j': 'Function words',
 'ils': 'Function words',
 'fusse': 'Function words',
 'furent': 'Function words',
 'du': 'Function words',
 'ses': 'Function words',
 'ont': 'Function words',
 'aurions': 'Function words',
 'pas': 'Function words',
 'étais': 'Function words',
 'soient': 'Function words',
 'avais': 'Function words',
 'il': 'Function words',
 'mon': 'Function words',
 'aie': 'Function words',
 'eurent': 'Function words',
 'étées': 'Function words',
 'étaient': 'Function words',
 'pour': 'Function words',
 'seriez': 'Function words',
 'seras': 'Function words',
 'auras': 'Function words',
 'aurons': 'Function words',
 'ne': 'Function words',
 'eût': 'Function words',
 'par': 'Function words',
 'serez': 'Function words',
 'n': 'Function words',
 'serait': 'Function words',
 'en': 'Function words',
 'aura': 'Function words',
 'l': 'Function words',
 'ce': 'Function words',
 'fus': 'Function words',
 'eus': 'Function words',
 'ayant': 'Function words',
 'eu': 'Function words',
 'ait': 'Function words',
 'ai': 'Function words',
 'eûmes': 'Function words',
 'ayez': 'Function words',
 'eusses': 'Function words',
 'eussent': 'Function words',
 'aies': 'Function words',
 'je': 'Function words',
 'au': 'Function words',
 'aient': 'Function words',
 'sera': 'Function words',
 'une': 'Function words',
 'fussent': 'Function words',
 'eussions': 'Function words',
 'seront': 'Function words',
 'auriez': 'Function words',
 'te': 'Function words',
 'eusse': 'Function words',
 'dans': 'Function words',
 'les': 'Function words',
 'mais': 'Function words',
 'ces': 'Function words',
 'on': 'Function words',
 'c': 'Function words',
 't': 'Function words',
 'soit': 'Function words',
 'soyez': 'Function words',
 'êtes': 'Function words',
 'aurait': 'Function words',
 'eut': 'Function words',
 'avez': 'Function words',
 'vos': 'Function words',
 'aviez': 'Function words',
 'le': 'Function words',
 'tes': 'Function words',
 'me': 'Function words',
 'moi': 'Function words',
 'sont': 'Function words',
 'ayantes': 'Function words',
 'et': 'Function words',
 'qu': 'Function words',
 'ta': 'Function words',
 'fussiez': 'Function words',
 'qui': 'Function words',
 'étés': 'Function words',
 'avec': 'Function words',
 'ma': 'Function words',
 'tu': 'Function words',
 'lui': 'Function words',
 'as': 'Function words',
 'nous': 'Function words',
 'étante': 'Function words',
 'vous': 'Function words',
 'serons': 'Function words',
 'sur': 'Function words',
 'de': 'Function words',
 'sa': 'Function words',
 'serai': 'Function words',
 'étantes': 'Function words',
 'eues': 'Function words',
 'elle': 'Function words',
 'serions': 'Function words',
 'seraient': 'Function words',
 'ou': 'Function words',
 'fûtes': 'Function words',
 'fusses': 'Function words',
 'avait': 'Function words',
 'eux': 'Function words',
 'toi': 'Function words',
 'ayants': 'Function words',
 'avions': 'Function words',
 'ayons': 'Function words',
 'se': 'Function words',
 'est': 'Function words',
 'étant': 'Function words',
 'leur': 'Function words',
 'suis': 'Function words',
 'ton': 'Function words',
 's': 'Function words',
 'soyons': 'Function words',
 'm': 'Function words',
 'fût': 'Function words',
 'ayante': 'Function words',
 'eûtes': 'Function words',
 'que': 'Function words',
 'd': 'Function words',
 'mes': 'Function words',
 'des': 'Function words',
 'la': 'Function words',
 'eue': 'Function words',
 'étants': 'Function words',
 'aurez': 'Function words',
 'à': 'Function words',
 'était': 'Function words',
 'son': 'Function words',
 'votre': 'Function words',
 'fussions': 'Function words',
 'étée': 'Function words',
 'sois': 'Function words',
 'notre': 'Function words',
 'es': 'Function words',
 'aurai': 'Function words',
 'serais': 'Function words',
 'un': 'Function words',
 'aurais': 'Function words',
 'y': 'Function words',
 'sommes': 'Function words',
 'fut': 'Function words',
 'auront': 'Function words',
 'auraient': 'Function words',
 'étions': 'Function words',
 'été': 'Function words',
 'même': 'Function words',
 'aux': 'Function words',
 'fûmes': 'Function words',
 'eussiez': 'Function words',
 'avons': 'Function words',
 'nos': 'Function words',
 'étiez': 'Function words'}
punctuation={"'": 'Punctuation',
    ':': 'Punctuation',
    ',': 'Punctuation',
    '_': 'Punctuation',
    '!': 'Punctuation',
    '?': 'Punctuation',
    ';': 'Punctuation',
    '.': 'Punctuation',
    '"': 'Punctuation',
    '(': 'Punctuation',
    ')': 'Punctuation',
    '-': 'Punctuation',
    '_SP': 'Punctuation',
    "''": 'Punctuation',
    '``': 'Punctuation',
    '$': 'Punctuation' ,
    "#": 'Punctuation',
    "@": 'Punctuation',
    "/": 'Punctuation'}
structural={'avg_w_len': 'Structural',
    'tot_short_w': 'Structural',
    'tot_digit': 'Structural',
    'tot_upper': 'Structural',
    'avg_s_len': 'Structural',
    'hapax': 'Structural',
    'dis': 'Structural',
    'syllable_count': 'Structural',
    'func_w_freq': 'Structural',
    'avg_w_freqc': 'Structural'}
indexes={'yules_K': 'Indexes',
    'shannon_entr': 'Indexes',
    'simposons_ind': 'Indexes',
    'flesh_ease': 'Indexes',
    'flesh_cincade': 'Indexes',
    'dale_call': 'Indexes',
    'gunnin_fox': 'Indexes'}

from style_approach.fr_extractor import postag_keys, function_word, punct
tag = dict(zip(postag_keys, ["TAG"]*len(postag_keys)))
function_words = dict(zip(function_word, ['Function words']*len(function_word)))
punctuation = dict(zip(punct, ["Punctuation"]*len(punct)))
map_features={**letters, **numbers, **ner, **tag, **function_words, **punctuation, **structural, **indexes}


def graph_format(features, test, output="agg") :
    features = features.set_index("id")

    res_df = features.copy()
    res_df = res_df.drop(['cluster'], axis=1)
    
    if output=="agg":
        for feature in tqdm.tqdm(res_df.columns):
            y=np.array(res_df[feature])
            y=(y - np.mean(y))/np.std(y)
            if np.isnan(y).any():
                continue
            res_df[feature] = y
            
        res_df = res_df.transpose()
        res_df['family']=res_df.index.map(map_features)
        res_df = res_df.groupby('family').mean()
        res_df = res_df.transpose()
        
    for feature in tqdm.tqdm(res_df.columns):
        
        y=np.array(res_df[feature])
        y=(y - np.mean(y))/np.std(y)

        if np.isnan(y).any():
            continue
        res_df[feature] = y
        
    
    res_df = res_df.join(features.cluster)
    
    final_df = pd.DataFrame()
    
    for cluster in pd.unique(res_df.cluster) :
        cluster_i_df = res_df.loc[res_df.cluster == cluster]
        test_df = []
        for feat in res_df.columns:
            if test == "t-test":
                test_df.append(stats.ttest_ind(res_df[feat], cluster_i_df[feat],equal_var=False)[1])
            if test == "ks-test":
                test_df.append(stats.kstest(res_df[feat], cluster_i_df[feat])[1])
        res_df_i = pd.DataFrame.from_dict({"mean":cluster_i_df.mean(),"std":cluster_i_df.std(), "test" : test_df}).transpose()
        res_df_i["cluster"] = [cluster, cluster, cluster]
        final_df = final_df.append(res_df_i.loc['mean'])
        final_df = final_df.append(res_df_i.loc['std'])
        final_df = final_df.append(res_df_i.loc['test'])
        
    return(final_df)


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
                return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

### Function to plot spyder chart given the feature values. ###
### If a value has a statistical score lower than "test_min", it will be highlighted ###
def style_spyder_charts(df_results, N_features,test_min = 1e-04, title="Clusters style evaluation"):


    N_clusters = len(df_results)/3
    N_graph = int(N_clusters//7) + 1

    N = N_features
    theta = radar_factory(N, frame='polygon')

    for i in range( N_graph) :
        df_results_i=df_results[df_results.index=='mean'].iloc[7*i:7*(i+1)]
        df_test_i=df_results[df_results.index=='test'].iloc[7*i:7*(i+1)]
        spoke_labels = list(df_results_i.drop('cluster', axis=1).columns)
    
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k'][:len(df_results_i)]
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
        # Plot the four cases from the example data on separate axes
        ax.set_rgrids([-1, -0.5, 0, 0.5, 1])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1),
                        horizontalalignment='center', verticalalignment='center')
        list_lines = []
        for d, color, test in zip(np.array(df_results_i.loc[:,spoke_labels]), colors, np.array(df_test_i.loc[:,spoke_labels])):
            l, = ax.plot(theta, d, color=color)
            list_lines.append(l)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
            ax.scatter(theta, d, color=color, s = 100*np.multiply(test < test_min,1))
        ax.set_varlabels(spoke_labels)
        print(list_lines)
        # add legend relative to top-left plot
        labels = tuple(df_results_i.cluster)
        legend = ax.legend(list_lines,labels, loc=(0.95, -0.1),
                                    labelspacing=0.1, fontsize = 'small')

    plt.show()