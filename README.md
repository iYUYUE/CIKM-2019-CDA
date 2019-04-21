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
- Train: python src/run.py train --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --baseline True

## DAMIC Model

with TF

- Train: python src/run.py train --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.6 --filters 200 --ld 0.15 --lr 0.0017 --tf 0.6 --lstm_hidden 900 --lstm_layers 2
- Test: python src/run.py test --model ./model/cflqxkjqlc/ --epoch 6 --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.6 --filters 200 --ld 0.15 --tf 0.6 --lstm_hidden 900 --lstm_layers 2
- Tune: python src/run.py tune --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog

without TF

- Train: python src/run.py train --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.4 --filters 200 --ld 0.2 --lr 0.0017 --lstm_hidden 1100 --lstm_layers 2
- Test: python src/run.py test --model ./model/rnmoaknnpi/ --epoch 6 --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.4 --filters 200 --ld 0.2 --lstm_hidden 1100 --lstm_layers 2


## DAMIC Wide-and-Deep Model
Train: python src/run.py train --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.4 --filters 240 --ld 0.15 --lr 0.0011 --tf 0.6 --lstm_hidden 440 --lstm_layers 4 --wd True
- Test: python src/run.py test --model ./model/qusfkublig/ --epoch 6 --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.4 --filters 240 --ld 0.15 --tf 0.6 --lstm_hidden 440 --lstm_layers 4 --wd True
- Tune: python src/run.py tune --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --batch_size 50 --wd True