import utils as util
import pandas as pd
import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for making plots with seaborn

import math  # for math
from scipy import stats


def getPrior(train):
  """
  On fait la somme de tous les éléments et on la divise par le nombre d'éléments pour obtenir la moyenne.
  On calcule la variance et  applique la formule pour les limites basse et haute. Estimation est la moyenne.
  1,96 est la valeur de la table qui correspond a Z0 = 0.95/2 + 0.5
  la valeur de retour est un dictionnaire
  """
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
    """
    l'estimation de la question précédente est supérieure
    à 0,5, nous pouvons donc supposer que la population entière a présenté la maladie cardiaque
    """
    return 1

  def statsOnDF(self, df):
    """
    nous parcourons la table et pour chaque tuple nous faisons une prévision et vérifions avec la
    valeur réelle. Nous incrémentons la valeur correspondant dans le dictionnaire. On calcule la
    précision et le rappel en utilisant la formule donnée.
    """
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
  """
  On prend toutes les valeurs uniques dans la target et toutes les valeurs uniques dans l'attribut
  en utilisant 2 sets. La clé du dictionnaire principal doit être target, donc pour chaque valeur on
  crée un nouveau dictionnaire avec toutes les valeurs de s2 commes cles, avec les valeurs correspondantes
  initialisées avec 0. On parcours le tableau et compte les apparitions de chaque attribut.
  A la fin on divise par la taille du tableau pour trouver la probabilité de chacun.
  """
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
  """
  On prend toutes les valeurs uniques dans la target et toutes les valeurs uniques dans l'attribut
  en utilisant 2 sets. La clé du dictionnaire principal doit être l'attribut, donc pour chaque valeur on
  crée un nouveau dictionnaire avec toutes les valeurs de s1 commes cles, avec les valeurs correspondantes
  initialisées avec 0. On parcours le tableau et compte les apparitions de chaque valeur de target.
  A la fin on divise par la taille du tableau pour trouver la probabilité de chacun.
  """
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
    """
    On calcule P2Dl un seul fois et
    on le garde dans la mémoire
    """
    self.attr = attr
    self.P2Dl = P2D_l(df, attr)


  def estimClass(self, attrs):
    """
    Les valeurs possible pour taget sont 1 ou 0. Donc on access les champs corespondants
    à la valeur de l'attribut pour obtenir la probabilité. 
    On compare les 2 valeurs et choisit le plus grand. En cas d'égalité choisir 0.
    """
    attribut = attrs[self.attr]
    val0 = self.P2Dl[0][attribut]
    val1 = self.P2Dl[1][attribut]

    if val0 < val1:
      return 1

    return 0


class MAP2DClassifier(APrioriClassifier):
  def __init__(self, df, attr):
    """
    On calcule P2Dl un seul fois et
    on le garde dans la mémoire
    """
    self.attr = attr
    self.P2Dp = P2D_p(df, attr)


  def estimClass(self, attrs):
    """
    Contrairement au cas précédent, les clés sont inversées, on accède donc aux éléments
    nécessaires en basculant les index. On compare les 2 valeurs et choisit le plus grand.
    En cas d'égalité choisir 0.
    """
    attribut = attrs[self.attr]
    val0 = self.P2Dp[attribut][0]
    val1 = self.P2Dp[attribut][1]

    if val0 < val1:
      return 1

    return 0

def nbParams(df, attributes=None):
  """
  Nous pouvons considérer ces dictionnaires imbriqués comme un arbre de recherche.
  Les probabilités sont les feuilles. Leur nombre est le produit de toutes les dimensions
  de chaque attribut. Nous devons également calculer la taille des noeuds intermédiaires.
  La dimension globale est la somme des taille(feuilles) et des taille(noeuds)
  Un dictionnaire vide a aussi une taille, mais dans ce cas nous le considérons 0.
  """

  if attributes == None:
    nb_attr = len(list(df))
    attributes = list(df)
  else:
    nb_attr = len(attributes)


  list_dimensions = []
  for attribute in attributes:
    s = set()
    for elem in df[attribute]:
      s.add(elem)
    list_dimensions.append(len(list(s)))

  total_attributes = np.prod(list_dimensions)
  total_size_feuilles = total_attributes * 8

  size_noeuds = 0
  dim_target = list_dimensions.pop(0) # eliminer dimensions de target
  list_dimensions = reversed(list_dimensions) # l'arbre branche premierement par le dernier attribut
  previous = 1
  size_of_int = 4 # octets
  
  for dimension in list_dimensions:
    size_noeuds += dimension * previous * size_of_int 
    previous *= dimension
  size_noeuds += previous * dim_target * size_of_int # the last row of nodes 


  total_size = size_noeuds + total_size_feuilles

  print(f"Number of attributes is {nb_attr}. Dimension in octets is {total_size}.")


def nbParamsIndep(df, attributes=None):
  """
  Chaque attribut est indépendant des autres. Nous pouvons les traiter séparément et à la
  fin faire la somme des dimensions. Pour chaque valeur d'un attribut, nous avons une probabilité, 
  donc, le nombre de clés de chaque table équivaut au nombre de probabilités.
  """

  if attributes == None:
    nb_attr = len(list(df))
    attributes = list(df)
  else:
    nb_attr = len(attributes)

  list_dimensions = []
  for attribute in attributes:
    s = set()
    for elem in df[attribute]:
      s.add(elem)
    list_dimensions.append(len(list(s)))

  size_of_int = 4
  size_of_float = 8
  total_attributes = np.sum(list_dimensions)
  total_size = total_attributes * size_of_float + total_attributes * size_of_int

  print(f"Number of attributes is {nb_attr}. Dimension in octets is {total_size}.")
