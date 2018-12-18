# DAMIC
Dialogue Act Modelling in Information-seeking Conversations

## Prerequisites
- MSDialog-Intent.json file (from MSdialog dataset; see https://ciir.cs.umass.edu/downloads/msdialog/ for information)
- Stanford NER (https://nlp.stanford.edu/software/CRF-NER.shtml)
- Python 3 with PyTorch, pandas and sklearn 


## Preprocessing *MSdialog-Intent.json* file
- Use *tabulate_msdialog.py* to convert the json file into csv format (the csv file is saved as msdialog.csv)
- Stemming and NER: use *msdialog_preprocessing_bow.py* to preprocess the converted csv file
- Use *collapse_utterance.py* to collapse similar utterances from the same speaker


## Markov chain
- Use *markov_chain.py* to produce Markov chain from original *MSDialog-Intent.json* file

## Baseline Model
- Train: python models/baseline.py train
- Test: python models/baseline.py test [model directory]

## DAMIC Model
- Train: python models/RNN.py train
- Test: python models/RNN.py test [model directory]
