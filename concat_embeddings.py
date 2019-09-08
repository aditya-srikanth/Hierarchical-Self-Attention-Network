import numpy as np 
import json 

vocab = {}
vectors = []
index = 0

with open('./glove/mapping.json', 'r') as f:
    vocab = json.load(f)


embedding = np.zeros((len(vocab), 400))

with open('glove/glove.42B.300d.txt', 'r' ) as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab: 
            vector = np.asarray( values[1:] )
            embedding[ vocab[ word ], :300 ] = vector  

print('glove done')

with open('glove/domain_embedding/restaurant_emb.vec','r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray( values[1:] )
        if word in vocab:
            embedding[vocab[ word ], 300 : ] = vector

with open('glove/concat_glove_restaurant.npz', 'wb') as f:
    np.save(f, embedding)



with open('glove/glove.42B.300d.txt', 'r' ) as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab: 
            vector = np.asarray( values[1:] )
            embedding[ vocab[ word ], :300 ] = vector  

print('glove done')

with open('glove/domain_embedding/laptop_emb.vec','r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray( values[1:] )
        if word in vocab:
            embedding[vocab[ word ]][ 300 : ] = vector

with open('glove/concat_glove_laptop.npz', 'wb') as f:
    np.save(f, embedding)
