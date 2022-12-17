
#####################################################################################################################
import pickle
import argparse
import traceback
import os
import pandas as pd
import numpy as np
import networkx as nx
import itertools
import sklearn
import torch


feature = "coexpression"
threshold = 137 ### note that it should be no less than 50 and no more than 500
#####################################################################################################################
if os.path.isfile('my' + str(feature) + 'nxlandgraph.edgelist'):
    G = nx.read_edgelist('my' + str(feature) + 'nxlandgraph.edgelist')
else:
    gene_graph = nx.read_edgelist('mycoexpressionnxgraph.edgelist')

    columns = ["gene"]
    #gene = set()
    all_exp_ids = [x for x in itertools.product(set(gene_graph.nodes))]
    all_exp_ids = pd.DataFrame(all_exp_ids, columns=columns)



    all_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in all_exp_ids.itertuples(name=None)]




    lm_id = []
    lm_probs_dict = {}
    infile = open('./datastore/bgedv2_GTEx_1000G_lm.txt')
    for line in infile:
        gene, id, probs = line.strip('\n').split('\t')
        lm_id.append(gene)



    print(gene_graph.subgraph(lm_id))

    G = gene_graph.subgraph(lm_id)

    print('edgesRAD9A')
    nx.write_edgelist(G, 'my' + str(feature) + 'nxlandgraph.edgelist')


lm_id = []
lm_probs_dict = {}
infile = open('./datastore/bgedv2_GTEx_1000G_lm.txt')
for line in infile:
    gene, id, probs = line.strip('\n').split('\t')
    lm_id.append(gene)


selected_edges = [(u,v) for u,v,e in G.edges(data=True) if e['coexpression'] > threshold]
print(len(selected_edges))

H = G.edge_subgraph(selected_edges)


boollist = []
for gene in lm_id:
    boollist.append(H.has_node(gene))

adjat = np.zeros([943,943])



adjacencymatrix = nx.to_numpy_matrix(H)
index = np.where(np.array(boollist)==True)



index = np.squeeze(index)

#adjat[index,:][:,index] = adjacencymatrix
matrix0 = np.zeros([len(index), 943])
matrix0[:, index] = adjacencymatrix
adjat[index, :] = matrix0



corrmat = torch.tensor(adjat).float()

torch.save(corrmat, 'stringDB1thres' + str(threshold) +'.pth')
print('success')
