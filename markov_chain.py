"""
Several references:

A good, comic tutorial to learn Markov Chain:
https://hackernoon.com/from-what-is-a-markov-model-to-here-is-how-markov-models-work-1ac5f4629b71

Tutorial (example code for using metworkx graphviz with pandas dataframe):
http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017



"""


import xmltodict, io, glob, json, os, re, random
from collections import defaultdict
import numpy as np
import random as rm
from itertools import chain
import pandas as pd
# import networkx.drawing.nx_pydot as gl
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
##matplotlib inline



ms_tags = ['CQ', 'FD', 'FQ', 'GG', 'IR', 'JK', 'NF', 'O', 'OQ', 'PA', 'PF', 'RQ']

ms_file = os.path.normpath(r'./data/msdialog/MSDialog-Intent.json')
ms_json =open(ms_file, 'r', encoding='utf8').read()
ms_dict = json.loads(ms_json)

# loading MSIntent json file
ms_intentlist = []
for secnum in ms_dict.keys():
    for utterance in ms_dict[secnum]["utterances"]:
        utt_tags = utterance["tags"].replace(" GG", "").replace("GG ", "")
        ms_intentlist.append(tuple([secnum,utterance["id"], utt_tags]))


# Markov model
# count dictionary
ct_dict = defaultdict(lambda: defaultdict(int))
rawct_dict = defaultdict(int)

START = "INITIAL"
END = "TERMINAL"
UNK = "<UNKNOWN>"

prev_tags = [START]
prev_sec = "0"

for ms in ms_intentlist:
    current_tags = ms[2].split(" ")
    if "" in current_tags:
        current_tags.remove("")
    current_sec = ms[0]
    if current_sec == prev_sec or prev_sec == "0":
        for j in current_tags:
            rawct_dict[j] += 1
            for i in prev_tags:
                ct_dict[i][j] += 1

    else:
        for i in prev_tags:
            ct_dict[i][END] += 1
        for j in current_tags:
            ct_dict[START][j] += 1
            rawct_dict[j] += 1

    prev_tags = current_tags
    prev_sec = current_sec





# create state space and initial state probabilities

states = ms_tags
pi = [1] + [0]*(len(ms_tags)-1)
state_space = pd.Series(pi, index=states, name='states')
print(state_space)
print(state_space.sum())

# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states



# convert counts into probs
prob_df = pd.DataFrame(columns=states+["TERMINAL"], index=states+["INITIAL"])

for k1 in ct_dict.keys():
    for k2 in ct_dict[k1].keys():
        # if (k1 in ms_tags+["TERMINAL", "INITIAL"]) and (k2 in ms_tags+["TERMINAL", "INITIAL"]):
        prob_df.loc [k1, k2] = round(ct_dict[k1][k2]/sum(ct_dict[k1].values()), 4)


prob_df.to_csv('markov_chain_prob.csv')



"""
The following codes generate the not-so-pretty flow figure in our previous Project Specification,
I chose to include a table in this final report.

q = prob_df.values
print('\n')
print(q, q.shape)
print('\n')
print(prob_df.sum(axis=1))

from pprint import pprint

# create a function that maps transition probability dataframe
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            # only for p>0.10:
            if Q.loc[idx,col] > 0.1:
                edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(prob_df)
pprint(edges_wts)

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states)
print('Nodes:\n')
print(G.nodes())
print('\n')

# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]

    # only for prob > 0.10
    if v>0.1:
        G.add_edge(tmp_origin, tmp_destination, label=v)
print('Edges:')
pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'markov.dot')
plt.show()

"""

