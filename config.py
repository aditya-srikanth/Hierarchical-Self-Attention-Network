sep = '\t'
PAD = 3

# file paths 
dataset_path = './datasets/Laptops_Train.xml'
test_dataset_path = './datasets/Laptops_Test.xml'
word_embedding_path = './glove/domain_embedding/laptop_emb.vec'
# word_embedding_path = './glove/glove.6B.100d.txt'
# embedding_save_path = './glove/embedding_matrix.npz'
# embedding_save_path = './glove/laptop_matrix.npy'
embedding_save_path = './glove/concat_glove_laptop.npz'

model_save_path = './model_weights/Laptop_Fusion_Attention_CRF_v2_10_fold_glove_domain_concat_embedding.pt'
save_stats_path = './results/Laptop_Fusion_Attention_CRF_v2_10_fold_glove_domain_concat_embedding.tsv'

bio_dict = { 'B': 2, 'I':1, 'O':0 }
POS_MAP =   {   '$': 0, '\'\'': 1, ',': 2, '-LRB-': 3, '-RRB-': 4, '.': 5, ':': 6, 'ADD': 7, 'AFX': 8, 'CC': 9, 
                'CD': 10, 'DT': 11, 'EX': 12, 'FW': 13, 'HYPH': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 
                'JJS': 18, 'LS': 19, 'MD': 20, 'NFP': 21, 'NN': 22, 'NNP': 23, 'NNPS': 24, 'NNS': 25, 
                'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33, 
                'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 
                'VBZ': 42, 'WDT': 43, 'WP': 44, 'WP$': 45, 'WRB': 46, 'XX': 47, '_SP': 48, '``': 49
            }

device = 'cuda'
# device = 'cpu'
num_dataset_workers = 0
rnn_model = 'lstm'


# hyper parameters.
num_epochs = 100
batch_size = 64
word_embeding_dim = 200
hidden_dim = 50
num_layers = 2
bidirectiional = True
dropout = 0.3 if num_layers > 1 else 0
lr= 0.001
momentum = 0.01
weight_decay= 1e-5
use_crf = True

# will be updated when the dataset is processed
max_review_length = 85