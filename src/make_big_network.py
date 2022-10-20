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
from itertools import chain, combinations

def mean_degree_connectivity(graph_name):
    num_edges = 0
    for node in graph_name.iterNodes():
        num_edges += len(graph_name.edges(node))

    return num_edges/graph_name.number_of_nodes()

def perc_nodes_in_lcc(graph_name):
    lcc = nk.centrality.LocalClusteringCoefficient(graph_name)
    lcc.run()
    largest_cc = max(lcc.ranking(), key=itemgetter(1))
    lenght = len(largest_cc)
    return lenght/graph_name.numberOfNodes()

# def mean_shortest_path_lcc(graph_name):
#     S = [graph_name.subgraph(c).copy() for c in nx.connected_components(graph_name)]
#     comps = [len(max(nx.connected_components(i), key=len)) for i in S]
#     index_max = max(range(len(comps)), key=comps.__getitem__)
#     return nx.average_shortest_path_length(S[index_max])

with open(r"free_assoc_full.pickle", "rb") as input_file:
    free_assoc = pickle.load(input_file)
    
with open(r"word_emb_list_98.pickle", "rb") as input_file:
    word_emb_list = pickle.load(input_file)    
    
with open(r"visual_layer_vectors_full.pickle", "rb") as input_file:
    visual_list = pickle.load(input_file)        
    
with open(r"lancaster_full.pickle", "rb") as input_file:
    lancaster_list = pickle.load(input_file)    

print(len(free_assoc))
print(len(word_emb_list))
print(len(visual_list))
print(len(lancaster_list))

layers = {'free_assoc':free_assoc,
          'word_emb_list':word_emb_list,
          'visual_list':visual_list, 
          'lancaster_list':lancaster_list}

len(layers)




def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

all_subsets = powerset(layers.keys())
all_subsets.pop(0) # the first subset is the empty set
print(len(all_subsets))


for sub_layers in all_subsets:
    print(sub_layers)
    
    words = []
    for k in sub_layers:
        i = layers[k]
        for a,b in i:
            words.append(a)
            words.append(b)

    words = list(set(words)) 
    print(len(words))
    ind = {words[i]:i for i in range(len(words))}
    dni = {i:words[i] for i in range(len(words))}
    pickle.dump(dni, open( " ".join(sub_layers) + "_index_word_full.pickle", "wb" ) ) 

    print('creating graph')
    new_3 = nk.Graph(len(words))
    # new_3.add_nodes_from(words)
    
    print('adding edges')
    for k in sub_layers:
        print(k)
        for a,b in tqdm(layers[k]):
            new_3.addEdge(ind[a], ind[b])    
    pickle.dump(new_3, open( "full_adult.pickle", "wb" ) )   

    # G = nk.nxadapter.nk2nx(new_3)
    # idmap = {}
    # for id, u in tqdm(zip(G.nodes(), range(G.number_of_nodes()))):
    #     idmap[id] = u
    
    # pickle.dump(idmap, open( "full_adult_idmap.pickle", "wb" ) )   


             

    print('computing statistics')
    print(nk.overview(new_3))
    # print('k', np.mean(nk.centrality.DegreeCentrality(new_3).run().scores()))
    # print('k', mean_degree_connectivity(new_3))
    # print('a', nk.correlation.Assortativity(new_3, range(len(words))))
    # print('CC', nk.globals.clustering(new_3))
    

    # print('conn', perc_nodes_in_lcc(new_3))
    # print('d', mean_shortest_path_lcc(new_3))
    
    print()