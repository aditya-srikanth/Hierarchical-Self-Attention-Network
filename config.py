sep = '\t'

# file paths 
dataset_path = './datasets/Restaurants_Train.xml'
test_dataset_path = './datasets/Restaurants_Test.xml'
# word_embedding_path = './glove/glove.6B.300d.txt'
# word_embedding_path = './glove/glove.42B.300d.txt'
word_embedding_path = './glove/domain_embedding/restaurant_emb.vec'
embedding_save_path = './glove/embedding_matrix.npy'
model_save_path = './model_weights/PBAN.pt'



bio_dict = { 'O':0, 'B':1, 'I':2 }
device = 'cuda'
# device = 'cpu'

# hyper parameters.
batch_size = 32
num_dataset_workers = 2

word_embeding_dim = 100
hidden_dim = 200
bidirectiional = True
num_layers = 2
dropout = 0.5 if num_layers > 1 else 0


# will be updated when the dataset is processed
max_review_length = 100
max_aspect_length = 10 