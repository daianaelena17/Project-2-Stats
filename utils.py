# ATTENTION : CE FICHIER NE DOIS PAS ETRE MODIFIE ! 
import numpy as np

from IPython.display import Image
import pydotplus

import matplotlib
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for making plots with seaborn

import math  # for math


def getNthDict(df, n):
  return df[n:n + 1].to_dict(orient='records')[0]


def viewData(data, kde=True):
  x = 4
  y = math.ceil(len(data.keys()) / x)
  plt.figure(figsize=(x * 4, y * 2))
  for i, k in enumerate(data.keys()):
    ax = plt.subplot(x, y, i + 1, xticklabels=[])
    ax.set_title("Distribution of '{0}': {1} in [{2},{3}]".format(
        k, len(data[k].unique()), data[k].min(), data[k].max()))
    ax = sns.distplot(data[k], kde=kde and len(data[k].unique()) > 5)
    ax.set_xlabel("")


def discretizeData(data):
  newData = data.copy()
  for k in newData.keys():
    if len(newData[k].unique()) > 5:
      newData[k] = data.apply(lambda row: np.digitize(row[k], np.histogram_bin_edges(newData[k], bins="fd")),
                              axis=1)
  return newData


class AbstractClassifier:
  def ___init__(self):
    pass

  def estimClass(self, attrs):

    raise NotImplementedError

  def statsOnDF(self, df):

    raise NotImplementedError


__GRAPHPREAMBULE = 'digraph{margin="0,0";node [style=filled, color = black, fillcolor=lightgrey,fontsize=10,' \
                   'shape=box,margin=0.05,width=0,height=0];'


def drawGraphHorizontal(arcs):

  graph = pydotplus.graph_from_dot_data(__GRAPHPREAMBULE + 'rankdir=LR;' + arcs + '}')
  return Image(graph.create_png())


def drawGraph(arcs):

  graph = pydotplus.graph_from_dot_data(__GRAPHPREAMBULE + arcs + '}')
  return Image(graph.create_png())

