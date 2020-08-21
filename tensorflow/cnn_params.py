# can be either "rest" or "lap". "rest" for restaurant and "lap" is for 
# laptops.
data_set = "rest"
# can be true or false
use_pos = True
vectors = "fasttext_skip_rest"
vectors2 = "fasttext_skip_rest"

params = {
  "ann_size": 10,
  "batch_size": 64,
  "bi_ann": False, 
  "concat_pos": True,
  "crf": True,
  "dependencies": "./dependencies/{}_dependencies.npy".format(data_set),
  "dependencies_pos": "./dependencies/{}_dependencies_pos.npy".format(data_set),
  "dependencies_ner": "./dependencies/{}_dependencies_ner.npy".format(data_set),
  "dropout": 0.5, 
  "epochs": 30, 
  "escape_word": "<ESCAPEWORD>", 
  "hybrid": True,
  "lp": False, 
  "lr": 0.005,
  "lr_decay": 0.4,
  "lstm_size": 50,
  "nepoch_no_improv": 5,
  "max_pooling": True,
  "out_channels_1": 200,
  "out_channels_2": 25,
  "pos_ann": False,
  "pos_ann_size": 20, 
  "single_ann": True, 
  "small_pos": False, 
  "test_file": "./test_data/{}_test.xml".format(data_set), 
  "test_size": 0.1, 
  "tokenizer": False, 
  "train_file": "./train_data/{}_train.xml".format(data_set), 
  "use_additional_embeddings": False,
  "use_char_embeddings": False,
  "use_elman": False,
  "use_gru": False,
  "use_pos": use_pos, 
  "use_rnn": False,
  "use_text_dataset": False,
  "use_window": False,
  "use_window_rnn": False,
  "w_dim": 100,
  "w_dim_2": 100
}

params["raw_vectors"] = "{}_{}d.txt".format(vectors, params["w_dim"])
params["raw_vectors_2"] = "{}_{}d.txt".format(vectors2, params["w_dim_2"])

if use_pos:
  params["word_file"] = "./words/{}_pos_words.txt".format(data_set)
  params["char_file"] = "./chars/{}_pos_chars.txt".format(data_set)
  params["pos_file"] = "./pos_npy/{}_pos.npy".format(data_set)
  params["pos_test_file"] = "./pos_files/{}_test_pos.txt".format(data_set)
  params["pos_train_file"] = "./pos_files/{}_train_pos.txt".format(
    data_set)

  params["vector_file"] = "./vectors/pos_{}_{}_{}d.npy".format(vectors,
    data_set, params["w_dim"])
  params["vector_file_2"] = "./vectors/pos_{}_{}_{}d.npy".format(vectors2,
    data_set, params["w_dim_2"])
else:
  params["word_file"] = "./words/{}_words.txt".format(data_set)
  params["char_file"] = "./chars/{}_chars.txt".format(data_set)
  params["vector_file"] = "./vectors/{}_{}_{}d.npy".format(vectors,
  data_set, params["w_dim"])
  params["vector_file_2"] = "./vectors/pos_{}_{}_{}d.npy".format(vectors2,
    data_set, params["w_dim_2"])
