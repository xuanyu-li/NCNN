
# Python program to convert .tsv file to .csv file
# importing pandas library
import pandas as pd
import networkx as nx
import os
import csv
import pickle

import random
tsv_file='./datastore/string_9606_ENSG_ENSP_10_all_T.tsv'

# readinag given tsv file
csv_table=pd.read_table(tsv_file,sep='\t')

# converting tsv file into csv
csv_table.to_csv('./datastore/string_9606_ENSG_ENSP_10_all_T.csv',index=False)

# output
print("Successfully made csv file")

ensmap = {}


savefile = "./datastore/ensp_ensg_df.pkl"


f = open(savefile, 'rb')
df = pickle.load(f)
f.close()

for index, row in df.iterrows():
    ensmap[row['protein_id']] = row['gene_id']


print(" reading self.proteinlinks")
edges = pd.read_csv("./datastore/9606.protein.links.detailed.v11.5.txt", sep=' ')

selected_edges = edges["coexpression"] != 0

feature = "coexpression"
edgelist = edges[selected_edges][["protein1", "protein2", feature]].values.tolist()


edgelist = [[ensmap[edge[0][5:]], ensmap[edge[1][5:]], edge[2]] for edge in edgelist
            if edge[0][5:] in ensmap.keys() and edge[1][5:] in ensmap.keys()]



G = nx.Graph()
G.add_weighted_edges_from([tuple(x) for x in edgelist], weight='coexpression')



nx.write_edgelist(G, 'my'+str(feature)+'nxgraph.edgelist')

