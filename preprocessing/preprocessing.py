import csv, os, itertools, spacy, os, re, copy, argparse
import pandas as pd
from collections import Counter
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
from functools import reduce
import numpy as np
from scipy.spatial.distance import dice
from itertools import chain
from statistics import stdev, mean
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



# Argument parser
parser = argparse.ArgumentParser(description='input parameters')

parser.add_argument('--outputcsvfile', '-o', action='store', dest='outputcsvfile', help='Output dataset csv file name (*.csv)')
parser.add_argument('--outputbowfile', '-b', action='store', dest='outputbowfile', help='Output BOW tab file name (*.tab)')
parser.add_argument('--length', '-l', default='10020', action='store', dest='length', help='Length of input file to read from')


parser.add_argument('--removestopwords', '-s', action='store_true', dest='removestopwords', help='specify if you would like to remove stopwords.')
parser.add_argument('--combine5W1H', '-w', action='store_true', dest='combine5W1H', help='specify if you would like to combine wh-words, what, when. why, who, where, how.')
parser.add_argument('--removeGGJKO', '-g', action='store_true', dest='removeGGJKO', help='specify if you would like to remove GG JK O when co-occurring with other DAs.')
parser.add_argument('--removepuncts', '-p', action='store_true', dest='removepuncts', help='specify if you would like to remove punctuations including ? !')
parser.add_argument('--bowfreqtype', '-f', action='store', dest='bowfreqtype', help='for BOW document frequency, choose DIALOGUE or TURN')
parser.add_argument('--bowthresholdtype', '-t', action='store', dest='bowthresholdtype', help='for BOW threshold type, choose DIALOGUE or TURN')
parser.add_argument('--bowthresholdnumber', '-n', action='store', default='3', dest='bowthresholdnumber', help='for BOW threshold minimum number (>=), default is 3.')
parser.add_argument('--bowminumumtokencount', '-m', action='store', default='5', dest='bowminumumtokencount', help='for BOW minimum total token count(>=), default is 5.')
parser.add_argument('--collapse', '-c', action='store_true', dest='collapse', help='collapse utterances from same speaker that pass a similarity threshold')



args = parser.parse_args()


# change to data directory
os.chdir(os.path.normpath(r'./../data/'))

ms_csvpath =  os.path.normpath(r'msdialog.csv')
df = pd.read_csv(ms_csvpath)


# output dataset (*.csv) and BOW (*.tab) file names
ms_csvpath = args.outputcsvfile
ms_bowpath = args.outputbowfile



# Stop words selected from NLTK, stop_words (excluding 5W1H)

all_stop_words = ['any', 'such', "weren't", 'now', 'just', 'myself',  'only', 'above', 'an', 'over', 'on', "you'd", 'yourselves', 'wasn',  'aren', 'the', 'same', 'ma', 'couldn', "hasn't", 'after', "can't", "how's", "mustn't", 'through', 'as', "that'll", 'once', 'wouldn', 'd', 'in', 'that', "aren't", 'i', 'while', 'itself', 'against', 'at', 'him', 'not', 'herself', "i've", "he's", "they'd", "hadn't", 'this', 've', 'these', "he'd", 'being', 'its', 'than', "doesn't", 'more', 'up', 'to', 'will', 'ourselves', 'haven', 'yours', 'but', 'off', 'very', 'her', 's', 'is', 'during', 'm', "we've", 'from', 'did', "don't", 'me', 'theirs', 'so', 'them', "you've", 'he', 'further', 'having', 'themselves', 'can', 'each', 'out', 'his', 'until', 'nor',  'would', "you'll", 'few', 'most', 'if', "isn't", 'be', 'had', 'between', 'could', 'we', 't', 'won', 'are', 'other', "she'd", 'their', "he'll", 'into', "mightn't", "didn't", 'then', 'should', 'doing',  'here', 'was', 'too', "we're", 'shan', "she's", "haven't", 'they', 'of', 'she', 'before', 'there', 'no', 'about', "i'd", 'ain', "we'd", "you're", "there's", 'a', "we'll", 'our', 'down', 'ought', 'were', "couldn't", 'own', 'y', 'both', "i'll", 'o', 'hasn', 're',  'have', "let's", 'cannot', 'under', 'needn', 'with', "wouldn't", 'shouldn', "won't", 'how', 'it', 'doesn', "shouldn't", "that's", 'again', "it's", 'am', 'my', 'hadn', "i'm", "should've", 'because', "she'll",  'by', 'do', "they're", "wasn't", 'weren',  "here's", "shan't", 'been', 'or', "they've", 'has', 'ours', 'all', 'isn', 'mustn', 'for', "needn't", 'those', 'yourself', 'mightn', 'does', 'don', 'didn', 'below', "they'll", 'himself', 'some', 'hers', 'll', 'your', 'and', 'you']

whwords = ["what's", 'whom',"when's",'which','where', "why's", "where's",'who','when', "who's",'what', 'why']

nertypes = ['NUM', 'ORGANIZATION', 'PERSON', 'URL', 'LOCATION', 'EMAIL', 'WHWORD', 'TECH']

ms_tags = ['CQ', 'FD', 'FQ', 'GG', 'IR', 'JK', 'NF', 'O', 'OQ', 'PA', 'PF', 'RQ']




# Stemmer & tagger
stemmer = PorterStemmer()
tagger = StanfordNERTagger(
    os.path.normpath('/Users/loganpeng/Dropbox/Parsers/stanford-ner-2017-06-09/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz'),
    os.path.normpath('/Users/loganpeng/Dropbox/Parsers/stanford-ner-2017-06-09/stanford-ner-2017-06-09/stanford-ner.jar'))



conversation_numbers = df['conversation_no'].tolist()
utterances = df['utterance'].tolist()
user_id = df['user_id'].tolist()
actor_type = df['actor_type'].tolist()




## add Begin, In, and End of a dialogue
all_utterance_status = []

length = int(args.length)


# adding B, I, E tag to the csv file
prev_conv = ''
for i, cnum in enumerate(conversation_numbers[:length]):

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




# Text Preprocessing
print('o Text preprocessing')

all_entitied_utterances = []
all_entitied_utterance_lists = []

tokenized_utterance = []

for idx, u in enumerate(utterances[:length]):

    # replace by hyper-tags
    text = re.sub(r'(http|file)[^\s]+', ' URL ', u)
    text = re.sub(r'www\.[^\s]+', ' URL ', text)
    text = re.sub(r'[^s]*[/\\><][^s]*[/\\><][^s]*', ' URL ', text)
    text = re.sub(r'[^\s]+\.(com|org|io|net|gov)[^\s]+', ' URL ', text)

    text = re.sub(r'[^s]*[-+=][^s]*[-+=][^s]*', ' TECH ', text)

    text = re.sub(r'[^\s]+@[^\s]+\.[^\s]+', ' EMAIL ', text)
    text = re.sub(r'\d+', ' NUM ', text)

    # internet languages
    text = re.sub(r'/', ' ', text)


    # tokenize
    tokenized_utterance.append(word_tokenize(text))




## NER
print('o NER processing')

classified_paragraphs_list = tagger.tag_sents(list(tokenized_utterance))

for i in classified_paragraphs_list:
    ner_processed = []
    for x in i:
        if x[1] in nertypes:
            ner_processed.append(x[1])
        elif x[0] in nertypes:
            ner_processed.append(x[0])
        else:
            ner_processed.append(stemmer.stem(x[0]))


    #  argparse arg:  remove punctuations
    if args.removepuncts:
        ner_processed = list(filter(lambda x: re.findall(r'^[^A-Za-z0-9\s]+$', x) == [], ner_processed))

    # remove redundant NER tags:
    for j in range(len(ner_processed)-1):
        if ner_processed[j] == ner_processed[j+1] and ner_processed[j] in nertypes:
            ner_processed[j] = ''
    ner_processed = list(filter(lambda x: x != '', ner_processed))


    # argparse arg: remove stop words
    if args.removestopwords:
        ner_processed = list(filter(lambda x: x not in all_stop_words, ner_processed))



    # argparse arg: replace 5W1H by supertag
    if args.combine5W1H:
        ner_processed = ['WHWORD' if x in whwords else x for x in ner_processed]


    # filter empty strings
    ner_processed = [x.strip() for x in ner_processed]

    all_entitied_utterance_lists.append(ner_processed)
    all_entitied_utterances.append(" ".join(ner_processed))




## Analyze DA tags
# remove redundant non-content tags: GG, JK, O
# argparser args: removeGGJKO  First remove O, then JK then GG based on their frequency
if args.removeGGJKO:
    new_tags = df['tags'].tolist()
    all_combos = []
    for t_id, t in enumerate(new_tags):
        l_combo = list(set(t.split()))

        if 'O' in l_combo and len(l_combo) > 1:
            l_combo.remove('O')
        if 'JK' in l_combo and len(l_combo) > 1:
            l_combo.remove('JK')
        if 'GG' in l_combo and len(l_combo) > 1:
            l_combo.remove('GG')

        all_combos.append(" ".join(l_combo))

        # df.at[t_id, "tags"] = " ".join(l_combo)
    df['tags'] = all_combos





## save (re-write) preprocessed utterances to the dataframe
print('o Saving pre-processed utterances to dataframe')

df.loc[0:length-1,'utterance'] = all_entitied_utterances
df['utterance_status'] = np.nan
df.loc[0:length-1,'utterance_status'] = all_utterance_status




## collapsing section here
tags = df['tags'].tolist()


def ret_index(li, s):
    if s in li:
        return li.index(s)
    else:
        return -1


def str2vector(li, str, norm):
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


def collapse():

    all_tags = []
    all_user_id = []
    all_actor_type = []
    all_utterances = []

    for i in range(len(all_entitied_utterances)):
        # for i in range(100):
        if all_utterance_status[i] == "B":
            dialog_user_id = [user_id[i]]
            dialog_tags = [tags[i]]
            dialog_actor_type = [actor_type[i]]
            dialog_utterances = [all_entitied_utterances[i]]

        else:
            dialog_tags.append(tags[i])
            dialog_user_id.append(user_id[i])
            dialog_actor_type.append(actor_type[i])
            dialog_utterances.append(all_entitied_utterances[i])

            if all_utterance_status[i] == 'E':
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
            if i2 != 0:
                prev_tags = str2vector(ms_tags, all_tags[i1][i2 - 1], False)
                prev_id = all_user_id[i1][i2 - 1]
                prev_type = all_actor_type[i1][i2 - 1]

                curr_tags = str2vector(ms_tags, c2, False)
                curr_id = all_user_id[i1][i2]
                curr_type = all_actor_type[i1][i2]

                curr_dice = dice(prev_tags, curr_tags)
                dices.append(curr_dice)
                if curr_id == prev_id and curr_type == prev_type:
                    if curr_type == "Agent":
                        all_agent_dices.append(curr_dice)
                    else:
                        all_user_dices.append(curr_dice)
            else:
                dices.append("NA")

    m = mean(list(filter(lambda x: x != "NA", dices)))
    sd = stdev(list(filter(lambda x: x != "NA", dices)))

    m_user = mean(list(filter(lambda x: x != "NA", all_user_dices)))
    m_agent = mean(list(filter(lambda x: x != "NA", all_agent_dices)))

    sd_user = stdev(list(filter(lambda x: x != "NA", all_user_dices)))
    sd_agent = stdev(list(filter(lambda x: x != "NA", all_agent_dices)))

    # find indexes of the rows that should be collapsed (similarity <= mean (user or agent specifically)
    collapse_index = []
    for id_d, d in enumerate(dices):
        if d != "NA":
            prev_type = actor_type[id_d - 1]
            prev_id = user_id[id_d - 1]
            curr_type = actor_type[id_d]
            curr_id = user_id[id_d]

            if curr_type == prev_type and curr_id == prev_id:
                if curr_type == "Agent":
                    if d <= m_agent:
                        collapse_index.append(id_d)
                else:
                    if d <= m_user:
                        collapse_index.append(id_d)

    for i in collapse_index:
        df.at[i - 1, "utterance"] = df.at[i - 1, "utterance"] + " " + df.at[i, "utterance"]

        # new combined DA tags
        l_combo = list(set(df.at[i - 1, "tags"].split() + df.at[i, "tags"].split()))

        df.at[i - 1, "tags"] = " ".join(l_combo)

    return df.drop(collapse_index)


## collapse function above

if args.collapse:
    print('o Collapsing utterances')
    df = collapse()

### end of collapsing

#
# ## Analyze DA tags
# # remove redundant non-content tags: GG, JK, O
# # argparser args: removeGGJKO  First remove O, then JK then GG based on their frequency
# if args.removeGGJKO:
#     new_tags = df['tags'].tolist()
#     all_combos = []
#     for t_id, t in enumerate(new_tags):
#         l_combo = list(set(t.split()))
#
#         if 'O' in l_combo and len(l_combo) > 1:
#             l_combo.remove('O')
#         if 'JK' in l_combo and len(l_combo) > 1:
#             l_combo.remove('JK')
#         if 'GG' in l_combo and len(l_combo) > 1:
#             l_combo.remove('GG')
#
#         all_combos.append(" ".join(l_combo))
#
#         # df.at[t_id, "tags"] = " ".join(l_combo)
#     df['tags'] = all_combos


## Save pre-processed dataframe to csv file
print('o Saving pre-processed dataframe to csv file')
df.to_csv(ms_csvpath)




## Frequency analysis section
print('o Analyzing token & doc frequencies')

new_utterances = df['utterance'].tolist()
new_utterance_lists = [x.split() for x in new_utterances]
new_utterance_pos = df['utterance_pos'].tolist()


# tok freq per dialogue
dialog_c = []
for ut_id, ut in enumerate(new_utterance_lists):
    if new_utterance_pos[ut_id] == 1:
        if ut_id != 0:
            dialog_c.append(curr_c)
        curr_c = Counter(ut)
    elif ut_id == len(new_utterance_lists)-1:
        curr_c.update(ut)
        dialog_c.append(curr_c)
    else:
        curr_c.update(ut)


# tok freq per turn
turn_c = [Counter(x) for x in new_utterance_lists]


# total token freq (across all docs)
flat_new_utterance_list = [item for sublist in new_utterance_lists for item in sublist]
c = Counter(flat_new_utterance_list).most_common()


# find doc freq for a specific token (could be per dialogue or per turn)
def tokdocfreq(tok, ct):
    return sum(list(map(lambda x: int(tok in x), ct)))



# write to BOW file
# argparse args: bowfreqtype (bow frequency type, DIALOGUE or TURN).
# argparse args: bowthresholdtype (bow threshold type, DIALOGUE or TURN).
# argparse args: bowthresholdnumber (bow threshold minimum, >=).

# threshold type
if args.bowthresholdtype.lower().strip() == 'dialogue':
    thres_c = dialog_c
elif args.bowthresholdtype.lower().strip() == 'turn':
    thres_c = turn_c
else:
    print('Please enter either DIALOGUE or TURN for bowthresholdtype')

# BOW freq type
if args.bowfreqtype.lower().strip() == 'dialogue':
    freq_c = dialog_c
elif args.bowfreqtype.lower().strip() == 'turn':
    freq_c = turn_c
else:
    print('Please enter either DIALOGUE or TURN for bowfreqtype')



## save BOW to tab file
print('o Saving token & doc frequencies to BOW tab file')

with open(ms_bowpath, 'w') as f:
    for k,v in c:
        if tokdocfreq(k, thres_c)>=int(args.bowthresholdnumber) and v>=int(args.bowminumumtokencount):
            f.write( "{}\t{}\n".format(k,tokdocfreq(k, freq_c)))














print("THE END...")
