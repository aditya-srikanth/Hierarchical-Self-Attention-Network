sep = '\t'
PAD = 3

bio_dict = { 'B': 2, 'I':1, 'O':0 }
POS_MAP =   {   '$': 0, '\'\'': 1, ',': 2, '-LRB-': 3, '-RRB-': 4, '.': 5, ':': 6, 'ADD': 7, 'AFX': 8, 'CC': 9, 
                'CD': 10, 'DT': 11, 'EX': 12, 'FW': 13, 'HYPH': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 
                'JJS': 18, 'LS': 19, 'MD': 20, 'NFP': 21, 'NN': 22, 'NNP': 23, 'NNPS': 24, 'NNS': 25, 
                'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33, 
                'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 
                'VBZ': 42, 'WDT': 43, 'WP': 44, 'WP$': 45, 'WRB': 46, 'XX': 47, '_SP': 48, '``': 49
            }

device = 'cuda'
num_dataset_workers = 0
rnn_model = 'lstm'

# hyper parameters.
num_epochs = 200
lambda1=1
batch_size = 64
hidden_dim = 50
num_layers = 2
bidirectiional = True
dropout = 0.5
use_crf = True
use_pos = False
optimizer = 'adadelta' # adadelta, adagrad, adam, adamax, asgd, rmsprop, sgd
model = 'hsan' # lstm, attention_lstm, global_attention_lstm, hsan, decnn
dataset = 'rest' # rest, laptop
embedding = 'concat_rest' # concat_rest, concat_laptop, rest, laptop, glove_rest, glove_laptop

num_folds= 1

# will be updated when the dataset is processed
max_review_length = 85

if dataset == 'laptop':
    mapping_file = './embeddings/laptop_mapping.json'
    dataset_path = './datasets/Laptops_Train.xml'
    test_dataset_path = './datasets/Laptops_Test.xml'
    if num_folds > 1:
        model_save_path = './model_weights/' + dataset + '_k_fold_' + model + '_' + embedding +'.pt'
        save_stats_path = './results/' + dataset + '_k_fold_' + model + '_' + embedding +'.tsv'
    else:
        model_save_path = './model_weights/' + dataset + '_' + model + '_' + embedding +'.pt'
        save_stats_path = './results/' + dataset + '_' + model + '_' + embedding +'.tsv'
 
elif dataset == 'rest':
    mapping_file = './embeddings/restaurant_mapping.json'
    dataset_path = './datasets/Restaurants_Train.xml'
    test_dataset_path = './datasets/Restaurants_Test.xml'
    if num_folds > 1:
        model_save_path = './model_weights/' + dataset + '_k_fold_' + model + '_' + embedding +'.pt'
        save_stats_path = './results/' + dataset + '_k_fold_' + model + '_' + embedding +'.tsv'
    else:
        model_save_path = './model_weights/' + dataset + '_' + model + '_' + embedding +'.pt'
        save_stats_path = './results/' + dataset + '_' + model + '_' + embedding +'.tsv'

if embedding == 'concat_laptop':
    word_embeding_dim = 200
    word_embedding_path = './embeddings/domain_embedding/laptop_emb.vec'
    embedding_save_path = './embeddings/concat_glove_domain_laptop.npy' # concatenated embedding

elif embedding == 'concat_rest':
    word_embeding_dim = 200
    word_embedding_path = './embeddings/domain_embedding/restaurant_emb.vec'
    embedding_save_path = './embeddings/concat_glove_domain_restaurant.npy'

elif embedding == 'laptop':
    word_embeding_dim = 100
    word_embedding_path = './embeddings/domain_embedding/laptop_emb.vec'
    embedding_save_path = './embdeddings/laptop_matrix.npy' 

elif embedding == 'rest':
    word_embeding_dim = 100
    word_embedding_path = './embeddings/domain_embedding/restaurant_emb.vec'
    embedding_save_path = './embeddings/restaurant_matrix.npy'

elif embedding == 'glove_rest':
    word_embeding_dim = 100
    word_embedding_path = './embeddings/glove/glove.6B.100d.txt'
    embedding_save_path = './embeddings/glove/rest_glove_matrix.npy'

elif embedding == 'glove_laptop':
    word_embeding_dim = 100
    word_embedding_path = './embeddings/glove/glove.6B.100d.txt'
    embedding_save_path = './embeddings/glove/laptop_glove_matrix.npy'
