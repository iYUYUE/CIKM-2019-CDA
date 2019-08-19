# Modeling Long-Range Context for Concurrent Dialogue Acts Recognition
This is the implementation for the CRNN model proposed in the CIKM'19 paper [Modeling Long-Range Context for Concurrent Dialogue Acts Recognition](https://arxiv.org/abs/xxxx.xxxxx). This model imposes fewer restrictions on the structure of DAs and captures textual features from a wider context.

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
```
python src/run.py train --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --baseline True
```
## DAMIC Model

to train, test and tune the base model with Teacher Forcing

```
python src/run.py train --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.6 --filters 200 --ld 0.15 --lr 0.0017 --tf 0.6 --lstm_hidden 900 --lstm_layers 2

python src/run.py test --model ./model/cflqxkjqlc/ --epoch 6 --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.6 --filters 200 --ld 0.15 --tf 0.6 --lstm_hidden 900 --lstm_layers 2

python src/run.py tune --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog
```

to train, test and tune the model without Teacher Forcing

```
python src/run.py train --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.4 --filters 200 --ld 0.2 --lr 0.0017 --lstm_hidden 1100 --lstm_layers 2

python src/run.py test --model ./model/rnmoaknnpi/ --epoch 6 --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --cd 0.4 --filters 200 --ld 0.2 --lstm_hidden 1100 --lstm_layers 2

python src/run.py tune --data_file ./data/msdialog/cnn/embedding_collapsed_spacytokenized_msdialog --tf 0.6
```
