#!/usr/bin/env python

import sys

import numpy as np

import cmapPy.pandasGEXpress.parse_gctx

GTEx_GCTX = 'GTEx_RNASeq_RPKM_n2921x55993.gctx'
BGEDV2_LM_ID = 'bgedv2_GTEx_1000G_lm.txt'
BGEDV2_TG_ID = 'bgedv2_GTEx_1000G_tg.txt'

information_data = cmapPy.pandasGEXpress.parse_gctx.parse(GTEx_GCTX,
                                                          convert_neg_666=True, ridx=None, cidx=None,
                                                          row_meta_only=False,
                                                          col_meta_only=False, make_multiindex=False)

genes = information_data.data_df.index.to_list()
GTEx_genes = map(lambda x:x.split('.')[0], genes)

lm_id = []
infile = open(BGEDV2_LM_ID)
for line in infile:
    ID = line.strip('\n').split('\t')[0]
    lm_id.append(ID)

infile.close()
lm_idx = map(GTEx_genes.index, lm_id)

tg_id = []
infile = open(BGEDV2_TG_ID)
for line in infile:
    ID = line.strip('\n').split('\t')[0]
    tg_id.append(ID)

infile.close()
tg_idx = map(GTEx_genes.index, tg_id)
genes_idx = lm_idx + tg_idx
data = information_data.data_df.values[genes_idx, :].astype('float64')


np.save('GTEx_float64.npy', data)



    
    
    
    
