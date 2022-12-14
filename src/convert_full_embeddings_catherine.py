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
from datetime import datetime

"""
Run as a script because this dies very early when run in a jupyter notebook. 

TODO: confirm that NetworKit Node2Vec implementation is leveraging weighted edges (as opposed to alternative network implementation
where we have duplicate edges instead of weighted) 


- G : networkit.Graph
    The graph.
- P : float
    The ratio for returning to the previous node on a walk.
    For P > max(Q,1) it is less likely to sample an already-visited node in the following two steps.
    For P < min(Q,1) it is more likely to sample an already-visited node in the following two steps.
- Q : float
    The ratio for the direction of the next step
    For Q > 1 the random walk is biased towards nodes close to the previous one.
    For Q < 1 the random walk is biased towards nodes which are further away from the previous one. 
- L : int
    The walk length.
- N : int
    The number of walks per node.
- D: int
    The dimension of the calculated embedding. 


https://github.com/networkit/networkit/blob/d467f1c2bd13b0f9e4ca3937d257856398c234e9/networkit/embedding.pyx
"""

threshold = 90
with open(r"generated_layer_data/free_assoc_visual_list_lancaster_list_bert_word_emb_list_index_to_word_BERT_intersect.pickle", "rb") as input_file:
    index_to_word = pickle.load(input_file)

with open(fr"networkit_graphs/full_adult_intersect_bert_{threshold}_cos_thresh_nk_weighted.pickle", "rb") as input_file:
    net = pickle.load(input_file)

print('running node2vec')
p_val = 0.5
q_val = 1.0
n_walks = 10
walk_len = 20
dims = 128

workers=1 # to make consecutive runs reproducable (paired w/ seed)

vecs  = Node2Vec(net,P=p_val,Q=q_val,L=walk_len, N=n_walks, D=dims)
vecs.run()

print('getting features')
vecs = vecs.getFeatures()

print(f"# vecs: {len(vecs)}, # word mapping: {len(index_to_word)}")

file_date_info = datetime.utcnow().strftime("%m-%d_%H:%M")
filename = f"networkit_embeddings/vectors_full_{threshold}_p{p_val}_q{q_val}_nwalk{n_walks}_wlen{walk_len}_dims{dims}_weighted_{file_date_info}.pickle"
pickle.dump(vecs, open(filename, "wb"))
print(f"filename: {filename}")


zipped_vecs = zip(list(index_to_word.values()), vecs)

filename = f"networkit_embeddings/zipped_vectors_full_{threshold}_p{p_val}_q{q_val}_nwalk{n_walks}_wlen{walk_len}_dims{dims}_weighted_{file_date_info}.pickle"
pickle.dump(zipped_vecs, open(filename, "wb"))
      
print(f"filename: {filename}")
