### Readme for NCNN
#### Prerequisites
Python 3.

Pytorch 1.11

Pytorch-geometric

#### Data
he original data files are not provided within this codebase, as some of them require applying for access. Once you download all of them, please put them in this codebase.

##### GEO and GTEx
The GEO and GTEx data we used in our paper is collected by emailing (yil8@uci.edu). For those who want to have the data, please reach https://github.com/uci-cbcl/D-GEX for further information.

##### Gene graph data
The gene interaction graph data can be downloaded on https://string-db.org/cgi/download.pl?sessionId=qJO5wpaPqJC7&species_text=Homo+sapiens.



#### Model
The proposed method NCNN is implemented and tested on two dataset-GEO dataset and GTEx dataset with two python files:

geo_NCNN_table.py and GTE_x_NCNN_table.py


The GCN model is implemented and tested on two dataset-GEO dataset and GTEx dataset with two python files:

geo_gcn_table.py and GTE_x_gcn_table.py
