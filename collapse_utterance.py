import os
import pandas as pd
from scipy.spatial.distance import dice
from itertools import chain
from statistics import stdev, mean
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


ms_tags = ['CQ', 'FD', 'FQ', 'GG', 'IR', 'JK', 'NF', 'O', 'OQ', 'PA', 'PF', 'RQ']



ms_entitiedpath = os.path.normpath(r'./data/entitied_msdialog.csv')
ms_collapsedpath = os.path.normpath(r'./data/collapsed_msdialog.csv')

df = pd.read_csv(ms_entitiedpath)

conversation_numbers = df['conversation_no']
utterance_tags = df['tags']
utterances = df['utterance']
utterance_status = df['utterance_status'] # B I E

utterance_index = df['utterance_index']
subutterance_index = df['subutterance_index']
user_id = df['user_id']
actor_type = df['actor_type']



all_tags = []
all_user_id = []
all_actor_type = []
all_utterances = []

def ret_index (li, s):
    if s in li:
        return li.index(s)
    else:
        return -1

def str2vector (li, str, norm):
    count = [ret_index(li, s) for s in str.split()]
    ret = [0] * len(li)
    for c in count:
        if c >= 0:
            ret[c] += 1
    s = sum(ret)
    if norm and s > 0:
        return [r / s for r in ret]
    else:
        return ret



for i in range(len(utterances)):
# for i in range(100):
    if utterance_status[i] == "B":
        dialog_user_id = [user_id[i]]
        dialog_tags = [utterance_tags[i]]
        dialog_actor_type = [actor_type[i]]
        dialog_utterances = [utterances[i]]

    else:
        dialog_tags.append(utterance_tags[i])
        dialog_user_id.append(user_id[i])
        dialog_actor_type.append(actor_type[i])
        dialog_utterances.append(utterances[i])

        if utterance_status[i] == 'E':
            all_tags.append(dialog_tags)
            all_actor_type.append(dialog_actor_type)
            all_user_id.append(dialog_user_id)
            all_utterances.append(dialog_utterances)


# dice distance DSC for the current tags and previous tags

# all_dices are of length 10020
dices = []
all_user_dices = []
all_agent_dices = []

for i1, c1 in enumerate(all_tags):
    for i2, c2 in enumerate(c1):
        if i2!=0:
            prev_tags = str2vector(ms_tags, all_tags[i1][i2-1], False)
            prev_id = all_user_id[i1][i2-1]
            prev_type = all_actor_type[i1][i2-1]

            curr_tags = str2vector(ms_tags, c2, False)
            curr_id = all_user_id[i1][i2]
            curr_type = all_actor_type[i1][i2]

            curr_dice = dice(prev_tags, curr_tags)
            dices.append(curr_dice)
            if curr_id==prev_id and curr_type==prev_type:
                if curr_type=="Agent":
                    all_agent_dices.append(curr_dice)
                else:
                    all_user_dices.append(curr_dice)
        else:
            dices.append("NA")



m = mean(list(filter(lambda x: x!="NA", dices)))
sd = stdev(list(filter(lambda x: x!="NA", dices)))

m_user = mean(list(filter(lambda x: x!="NA", all_user_dices)))
m_agent = mean(list(filter(lambda x: x!="NA", all_agent_dices)))

sd_user = stdev(list(filter(lambda x: x!="NA", all_user_dices)))
sd_agent = stdev(list(filter(lambda x: x!="NA", all_agent_dices)))


# find indexes of the rows that should be collapsed (similarity <= mean (user or agent specifically)
collapse_index = []
for id_d, d in enumerate(dices):
    if d != "NA":
        prev_type = actor_type[id_d-1]
        prev_id = user_id[id_d-1]
        curr_type = actor_type[id_d]
        curr_id = user_id[id_d]

        if curr_type==prev_type and curr_id==prev_id:
            if curr_type == "Agent":
                if d <= m_agent:
                    collapse_index.append(id_d)
            else:
                if d <= m_user:
                    collapse_index.append(id_d)


for i in collapse_index:
    df.at[i-1, "utterance"] = df.at[i-1, "utterance"] + " \n " + df.at[i, "utterance"]
    df.at[i - 1, "tags"] =  " ".join(list(set(df.at[i - 1, "tags"].split() + df.at[i, "tags"].split())))


df = df.drop(collapse_index)

df.to_csv(ms_collapsedpath)



