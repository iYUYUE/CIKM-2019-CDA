# -*- coding: utf-8 -*-

import csv, json, sys, os
import numpy as np
from pandas.io.json import json_normalize


ms_tags = ['CQ', 'FD', 'FQ', 'GG', 'IR', 'JK', 'NF', 'O', 'OQ', 'PA', 'PF', 'RQ']

# Please change your path to read the MSDialog-Intent.json file from MSDialog corpus
ms_file = os.path.normpath(r'your/path/to/MSDialog-Intent.json')


# file to save tabulated csv file (from original json file)
ms_csvpath =  os.path.normpath(r'./data/msdialog.csv')

with open(ms_file) as data_file:
    data = json.load(data_file)

conversation_list = list(data.keys())

# add conversation number into data dictionary
for k,v in data.items():
    v['msdialog_filename'] = k
    v['conversation_no'] = k
    prev_user = ''
    utterance_index = 0

    for u in v['utterances']:
        curr_user = u['user_id']
        if curr_user != prev_user:
            utterance_index += 1
            subutterance_index = 1
        else:
            subutterance_index += 1
        prev_user = curr_user
        u['utterance_index'] = utterance_index
        u['subutterance_index'] = subutterance_index


df = json_normalize(data.values(), 'utterances', ['title', 'category', 'dialog_time', 'frequency', 'msdialog_filename', 'conversation_no'])

# add empty column and reindex to match Switchboard corpus
df["ptb_basename"] = np.nan
df["pos"] = np.nan
df["trees"] = np.nan
df["ptb_treenumbers"] = np.nan

# re-index df to match Switchboard corpus
df = df.reindex(columns=['msdialog_filename', 'ptb_basename', 'conversation_no', 'utterance_pos', 'tags', 'user_id',
                         'utterance_index', 'subutterance_index', 'utterance', 'pos',	'trees', 'ptb_treenumbers',
                         'actor_type', 'affiliation', 'id', 'is_answer', 'utterance_time', 'vote', 'title', 'category', 'dialog_time', 'frequency'
                         ])


df.to_csv(ms_csvpath)
print('Done!')


