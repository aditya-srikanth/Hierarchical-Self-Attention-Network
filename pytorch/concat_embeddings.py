import numpy as np 
import json 

from data_utils import Vocab

vocab = {}
vectors = []
index = 0

train_dataset = './datasets/Restaurants_Train.xml'
test_dataset = './datasets/Restaurants_Test.xml'
mapping_file = './embeddings/restaurant_mapping.json'
vocab = Vocab.from_files( [train_dataset, test_dataset], store= mapping_file ).get_vocab()

embedding = np.zeros((len(vocab), 200))

with open('embeddings/glove/glove.6B.100d.txt', 'r', encoding= 'utf-8' ) as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab: 
            vector = np.asarray( values[1:] )
            embedding[ vocab[ word ], :100 ] = vector  

print('glove done')

with open('embeddings/domain_embedding/restaurant_emb.vec','r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray( values[1:] )
        if word in vocab:
            embedding[vocab[ word ], 100 : ] = vector

with open('embeddings/concat_glove_domain_restaurant.npy', 'wb') as f:
    np.save(f, embedding)

print('saved restaurant glove embedding')

train_dataset = './datasets/Laptops_Train.xml'
test_dataset = './datasets/Laptops_Test.xml'
mapping_file = './embeddings/laptop_mapping.json'
vocab = Vocab.from_files( [train_dataset, test_dataset], store= mapping_file ).get_vocab()

embedding = np.zeros((len(vocab), 200))
with open('embeddings/glove/glove.6B.100d.txt', 'r', encoding='utf-8' ) as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab: 
            vector = np.asarray( values[1:] )
            embedding[ vocab[ word ], :100 ] = vector  

print('glove done')

with open('embeddings/domain_embedding/laptop_emb.vec','r', encoding= 'utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray( values[1:] )
        if word in vocab:
            embedding[vocab[ word ]][ 100 : ] = vector

with open('embeddings/concat_glove_domain_laptop.npy', 'wb') as f:
    np.save(f, embedding)

print('saved laptop glove embedding')