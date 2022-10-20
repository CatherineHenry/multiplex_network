
import networkit as nk
import torch
from networkx.algorithms import approximation
import numpy as np
from tqdm import tqdm_notebook as tqdm
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import pickle
from nltk.stem import WordNetLemmatizer
from operator import itemgetter
import collections
from networkit.embedding import Node2Vec 


dims = 128

with open(r"free_assoc word_emb_list visual_list lancaster_list_index_word_full.pickle", "rb") as input_file:
    dni = pickle.load(input_file) 

with open(r"full_adult.pickle", "rb") as input_file:
    net = pickle.load(input_file)

print('running node2vec')
vecs  = Node2Vec(net,P=1.0,Q=0.1,L=1, N=1, D=dims)
vecs.run()

print('getting features')
vecs = vecs.getFeatures()

print(len(vecs), len(dni))
vecs = zip(list(dni.keys()), vecs)


pickle.dump(vecs, open( "vectors_full_{}.pickle".format(dims), "wb" ) )  
