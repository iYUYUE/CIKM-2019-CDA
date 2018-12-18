import csv, os, itertools, spacy, os, re, copy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
from functools import reduce
import numpy as np



stemmer = PorterStemmer()
tagger = StanfordNERTagger(
    os.path.normpath('/Users/loganpeng/Dropbox/Parsers/stanford-ner-2017-06-09/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz'),
    os.path.normpath('/Users/loganpeng/Dropbox/Parsers/stanford-ner-2017-06-09/stanford-ner-2017-06-09/stanford-ner.jar'))



ms_csvpath =  os.path.normpath(r'./data/msdialog.csv')
df = pd.read_csv(ms_csvpath)
ms_entitiedcsvpath = os.path.normpath(r'./data/entitied_msdialog.csv')
ms_entitiedbowpath = os.path.normpath(r'./data/entitied_bow.tab')

conversation_numbers = df['conversation_no']
utterances = df['utterance']

all_utterance_status = []



length = len(utterances)


# adding B, I, E tag to the csv file
prev_conv = ''
for i, cnum in enumerate(conversation_numbers[:length]):
    print(i, sep=' ')

    try:
        if conversation_numbers[i-1] != conversation_numbers [i]:
            all_utterance_status.append('B')
        else:
            try:
                if conversation_numbers[i] != conversation_numbers[i + 1]:
                    all_utterance_status.append('E')
                else:
                    all_utterance_status.append('I')
            except:
                all_utterance_status.append('E')
    except:
        all_utterance_status.append('B')




all_entitied_utterances = []
all_entitied_utterance_lists = []


# Preprocessing
for idx, u in enumerate(utterances[:length]):
    print(idx, sep=' ')

    # replace by hyper-tags
    text = re.sub(r'http[^\s]+', 'URL', u)
    text = re.sub(r'www\.[^\s]+', 'URL', text)
    text = re.sub(r'[^\s]+@[^\s]+\.[^\s]+', 'EMAIL', text)
    text = re.sub(r'\d+', 'NUM', text)
    while re.findall(r'[\.,\?\(\)\[\]:/\!_\"]', text) != []:
        # text = re.sub(r'([\s])[\.,\?\(\)\[\]\:]', '\1', text)
        # text = re.sub(r'[\.,\?\(\)\[\]\:]([\s])', '\1', text)
        text = re.sub(r'[\.,\?\(\)\[\]:/\!_\"]', ' ', text)

    tokenized_utterance = []
    tokenized_utterance.append(word_tokenize(text))


    classified_paragraphs_list = tagger.tag_sents(list(tokenized_utterance))

    # replace NER tokens by their NER-type
    processed = [stemmer.stem(x[0]) if x[1]=='O' else x[1] for x in classified_paragraphs_list[0]]
    ner_processed = [x.lower() if x.upper()!=x else x for x in processed]

    # remove redundant NER tags:
    for i in range(len(ner_processed)):
        try:
            if ner_processed[i-1] == ner_processed[i] and i>0:
                ner_processed.pop(i)
        except:
            pass

    all_entitied_utterance_lists.append(ner_processed)
    all_entitied_utterances.append(" ".join(ner_processed))

flat_entitied_list = [item for sublist in all_entitied_utterance_lists for item in sublist]


c = Counter(flat_entitied_list).most_common()


# write to BOW file
with open(ms_entitiedbowpath, 'w') as f:
    for k,v in c:
        if v>=5:
            f.write( "{}\t{}\n".format(k,v))





# write new csv file (with stemming, entity replaced)
entitied_df = df.copy()

entitied_df.loc[0:length-1,'utterance'] = all_entitied_utterances
entitied_df['utterance_status'] = np.nan
entitied_df.loc[0:length-1,'utterance_status'] = all_utterance_status

entitied_df.to_csv(ms_entitiedcsvpath)

print("THE END...")
