import utils as util
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for making plots with seaborn

import math  # for math
from scipy import stats


def getPrior(train):
    sample = train.copy()
    estim = sample['target'].sum() / sample['target'].size
    var_list = sample['target'].tolist()
    sum_var = 0
    for i in var_list:
      sum_var += (i - estim) ** 2
    var = sum_var/sample['target'].size
    var_sqrt = math.sqrt(var)

    min = estim - 1.96 * var_sqrt/math.sqrt(sample['target'].size)
    max = estim + 1.96 * var_sqrt/math.sqrt(sample['target'].size)
    x = {}
    x['estimation'] = estim
    x['min5pourcent'] = min
    x['max5pourcent'] = max
    return x

class APrioriClassifier(util.AbstractClassifier):
  def ___init__(self):
    pass

  def estimClass(self, attrs):
    # x = dict(attrs)
    # print(getPrior(x))
    # print(attrs)
    return 1

  def statsOnDF(self, df):
    # VP : nombre d'individus avec target=1 et classe prévue=1
    # VN : nombre d'individus avec target=0 et classe prévue=0
    # FP : nombre d'individus avec target=0 et classe prévue=1
    # FN : nombre d'individus avec target=1 et classe prévue=0
    # précision
    # rappel
    x = {}
    x['VP'] = 0
    x['VN'] = 0
    x['FP'] = 0
    x['FN'] = 0
    x['precision'] = 0
    x['rappel'] = 0
    for t in df.itertuples():
      dic = t._asdict()
      v = dic['target']
      classePrevue = self.estimClass(dic)
      if v == 1 and classePrevue == 1:
        x['VP'] += 1
      elif v == 0 and classePrevue == 0:
        x['VN'] += 1
      elif v == 0 and classePrevue == 1:
        x['FP'] += 1
      else:
        x['FN'] += 1
    x['Precision'] = x['VP'] / (x['VP'] + x['FP'])
    x['Rappel'] = x['VP'] / (x['VP'] + x['FN'])
    return x



def P2D_l(df, attr):
  x = {}

  s1 = set()
  s2 = set()
  for i in df['target']:
    s1.add(i)

  for i in df[attr]:
    s2.add(i)

  for i in s1:
    x[i] = {}
    for j in s2:
      x[i][j] = 0

  size_target = df.groupby('target')[attr].count()

  for i in range(0, df.shape[0]):
    dic = util.getNthDict(df, i)
    x[dic['target']][dic[attr]] += 1

  for target, attribute in x.items():
    for key in attribute:
        attribute[key] /= size_target[target]

  return x

def P2D_p(df,attr):
  x = {}

  s1 = set()
  s2 = set()
  for i in df['target']:
    s1.add(i)

  for i in df[attr]:
    s2.add(i)

  for i in s2:
    x[i] = {}
    for j in s1:
      x[i][j] = 0

  size_target = df.groupby(attr)['target'].count()

  for i in range(0, df.shape[0]):
    dic = util.getNthDict(df, i)
    x[dic[attr]][dic['target']] += 1

  for attribute, target in x.items():
    for key in target:
        target[key] /= size_target[attribute]

  return x

class ML2DClassifier(APrioriClassifier):
  def __init__(self, df, attr):
    self.df = df
    self.attr = attr
    self.P2Dl = P2D_l(df, attr)


  def estimClass(self, attrs):
    attribut = attrs[self.attr]
    # print(attribut)
    # print(self.P2Dl)
    val0 = self.P2Dl[0][attribut]
    val1 = self.P2Dl[1][attribut]

    if val0 < val1:
      return 1

    return 0


class MAP2DClassifier(APrioriClassifier):
  def __init__(self, df, attr):
    self.df = df
    self.attr = attr
    self.P2Dp = P2D_p(df, attr)


  def estimClass(self, attrs):
    attribut = attrs[self.attr]
    # print(attribut)
    # print(self.P2Dp)
    val0 = self.P2Dp[attribut][0]
    val1 = self.P2Dp[attribut][1]

    if val0 < val1:
      return 1

    return 0

def nbParams(df, attributes=None):

  if attributes == None:
    nb_attr = len(list(df))
    attributes = list(df)
    # print(nb_attr)
  else:
    nb_attr = len(attributes)

  list_dimensions = []
  for attribute in attributes:
    s = set()
    for elem in df[attribute]:
      s.add(elem)
    list_dimensions.append(len(list(s)))

  total_attributes = np.prod(list_dimensions)
  total_size = total_attributes * 8
  # print(total_size)
  print(f"Number of attributes is {nb_attr}. Dimension in octets is {total_size}.")


def nbParamsIndep(df, attributes=None):

  if attributes == None:
    nb_attr = len(list(df))
    attributes = list(df)
    # print(nb_attr)
  else:
    nb_attr = len(attributes)

  list_dimensions = []
  for attribute in attributes:
    s = set()
    for elem in df[attribute]:
      s.add(elem)
    list_dimensions.append(len(list(s)))

  total_attributes = np.sum(list_dimensions)
  total_size = total_attributes * 8
  # print(total_size)
  print(f"Number of attributes is {nb_attr}. Dimension in octets is {total_size}.")
