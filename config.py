sep = '\t'

# file paths 
dataset_path = './datasets/Restaurants_Train.xml'
test_dataset_path = './datasets/Restaurants_Test.xml'
# word_embedding_path = './glove/glove.6B.300d.txt'
word_embedding_path = './glove/glove.6B.100d.txt'
embedding_save_path = './glove/concat_glove_restaurant.npz'
# embedding_save_path = './glove/embedding_matrix.npy'
model_save_path = './model_weights/Rest_ASE_Glove_attention_concat_embedding.pt'
save_stats_path = './results/Rest_ASE_Glove_attention_concat_embedding.tsv'

bio_dict = { 'O':0, 'B':1, 'I':2 }
device = 'cuda'
# device = 'cpu'

# hyper parameters.
num_epochs = 200
batch_size = 16
num_dataset_workers = 4

word_embeding_dim = 200
hidden_dim = 50
bidirectiional = True
num_layers = 2
dropout = 0.3 if num_layers > 1 else 0


# will be updated when the dataset is processed
max_review_length = 80