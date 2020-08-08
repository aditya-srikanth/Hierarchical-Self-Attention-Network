# Attention-Aspect-Extraction

## Prerequisites

spacy, pytorch, [pytorch-crf](https://pytorch-crf.readthedocs.io/en/stable/), numpy

## Installing
1. Clone the repository
2. Download glove embeddings (6B 100d) and paste them in embeddings/glove. Link: (http://nlp.stanford.edu/data/glove.6B.zip)
3. Paste domain embeddings in embedding/domain_embedding
4. Run concat_embeddings.py to generate glove-domain concactenated embeddings

## Running the tests
* Update configuration and run train.py

## Hyperparameters and configurations:
Present in config.py
### Defaults
* seperator between the processed dataset (stored as a .tsv file). Default: \t
* Padding token used for sequence labels as one-hot labels. Default: 3.
* BIO dictionary: used for generating sequence labels.
* POS MAP: for encoding POS tags. 
* device: cpu or cuda
* num_dataset_workers: for batching the dataset. Default 0.
* max_review_length = 85

### Hyper Parameters.
* rnn_model: lstm or gru. Default= lstm
* num_epochs: Default= 50
* batch_size: Default= 64
* hidden_dim: Default= 50
* num_layers: Default= 2
* bidirectiional: Default= True
* dropout: Default= 0.5
* use_crf: Default= True
* use_pos: Default= False
* optimizer: adadelta, adagrad, adam, adamax, asgd, rmsprop, sgd
* model: lstm, attention_lstm, global_attention_lstm, hsan, decnn
* dataset: rest, laptop
* embedding: concat_rest, concat_laptop, rest, laptop, glove_rest, glove_laptop
* num_folds: Default= 1

#### Hyperparameters for training HSAN
* rnn_model: lstm or gru. Default= lstm
* num_epochs: Default= 50
* batch_size: Default= 64
* hidden_dim: Default= 50
* num_layers: Default= 2
* bidirectiional: Default= True
* dropout: Default= 0.3
* use_crf: Default= True
* use_pos: Default= False
* optimizer: adadelta, adagrad, adam, adamax, asgd, rmsprop, sgd
* model: hsan
* dataset: rest OR laptop
* embedding: concat_rest, OR concat_laptop
* num_folds: Default= 1
