import os
import pandas as pd


ms_entitiedbowpath = os.path.normpath("./data/msdialog/entitied_msdialog_100.csv")


df = pd.read_csv(ms_entitiedbowpath)

conversation_numbers = df['conversation_no']
utterance_tags = df['tags']
utterances = df['utterance']
utterance_status = df['utterance_status']


all_utterances = []
all_tags = []

for i in range(len(utterances)):
    if utterance_status[i] == "B":
        dialog_utterances = [utterances[i]]
        dialog_tags = [utterance_tags[i]]

    else:
        dialog_utterances.append(utterances[i])
        dialog_tags.append(utterance_tags[i])
        if utterance_status[i] == 'E':
            all_utterances.append(dialog_utterances)
            all_tags.append(dialog_tags)



