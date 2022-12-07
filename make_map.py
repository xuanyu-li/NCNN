
# Python program to convert .tsv file to .csv file
# importing pandas library
import pandas as pd
import networkx as nx
import os
import csv
import pickle
from gtfparse import read_gtf
import random
tsv_file='string_9606_ENSG_ENSP_10_all_T.tsv'

# readinag given tsv file
csv_table=pd.read_table(tsv_file,sep='\t')

# converting tsv file into csv
csv_table.to_csv('string_9606_ENSG_ENSP_10_all_T.csv',index=False)

# output
print("Successfully made csv file")

ensmap = {}
# for index, row in df.iterrows():
#     if row['gene_id'] in ensg_map.keys():
#         ensmap[row['protein_id']] = ensg_map[row['gene_id']]

from gtfparse import read_gtf

savefile = "./data" + "/datastore/ensp_ensg_df.pkl"

# If df is already stored, return the corresponding dictionary
if os.path.isfile(savefile):
    f = open(savefile, 'rb')
    df = pickle.load(f)
    f.close()
else:
    df = read_gtf("./data" + "/datastore/Homo_sapiens.GRCh38.95.gtf")
    df = df[df['protein_id'] != ''][['gene_id', 'protein_id']].drop_duplicates()
    df.to_pickle(savefile)

for index, row in df.iterrows():
    ensmap[row['protein_id']] = row['gene_id']


print(" reading self.proteinlinks")
edges = pd.read_csv("9606.protein.links.detailed.v11.5.txt", sep=' ')

selected_edges = edges["coexpression"] != 0

feature = "coexpression"
edgelist = edges[selected_edges][["protein1", "protein2", feature]].values.tolist()


edgelist = [[ensmap[edge[0][5:]], ensmap[edge[1][5:]], edge[2]] for edge in edgelist
            if edge[0][5:] in ensmap.keys() and edge[1][5:] in ensmap.keys()]



G = nx.Graph()
G.add_weighted_edges_from([tuple(x) for x in edgelist], weight='coexpression')



nx.write_edgelist(G, 'my'+str(feature)+'nxgraph.edgelist')

