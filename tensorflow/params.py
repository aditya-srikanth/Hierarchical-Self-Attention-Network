# can be either "lap" or "rest". "rest" for restaurant and "lap" is for
# laptops.
data_set = "rest"
# can be true or false
use_pos = True
# can be amazon or glove (or fasttext if we incorporate that too)
vectors = "fasttext_skip_rest"
vectors2 = "fasttext_cbow_rest"

params = {
  "c_dim": 10,
  "batch_size": 64,
  "bi_ann": False,
  "bidirectional": True,
  "concat_pos": True,
  "crf": True,
  "dropout": 0.5,
  "epochs": 30,
  "escape_word": "<ESCAPEWORD>",
  "lr": 0.005,
  "lr_decay": 0.9,
  "lstm_ann_count": 40,
  "multi_rnn": False,
  "nepoch_no_improv": 5,
  "pos_ann": False,
  "pos_ann_count": 20,
  "raw_vectors": "./{}_{}d.txt",
  "single_ann": False,
  "small_pos": False,
  "test_file": "./test_data/{}_test.xml".format(data_set),
  "test_size": 0.1,
  "tokenizer": False,
  "train_file": "./train_data/{}_train.xml".format(data_set),
  "use_char_embeddings": True,
  "use_elman": False,
  "use_gru": False,
  "use_pos": use_pos,
  "use_rnn": False,
  "use_window": False,
  "use_text_dataset": False,
  "w_dim": 100,
  "w_dim_2": 100
}
params["raw_vectors"] = "./{}_{}d.txt".format(vectors, params["w_dim"])
params["raw_vectors_2"] = "./{}_{}d.txt".format(vectors2, params["w_dim_2"])
if use_pos:
  params["word_file"] = "./words/{}_pos_words.txt".format(data_set)
  params["char_file"] = "./chars/{}_pos_chars.txt".format(data_set)
  params["pos_file"] = "./pos_npy/{}_pos.npy".format(data_set)
  params["pos_test_file"] = "./pos_files/{}_test_pos.txt".format(data_set)
  params["pos_train_file"] = "./pos_files/{}_train_pos.txt".format(
    data_set)
  params["vector_file"] = "./vectors/pos_{}_{}_{}d.npy".format(vectors, data_set,
    params["w_dim"])
  params["vector_file_2"] = "./vectors/pos_{}_{}_{}d.npy".format(vectors2, data_set,
    params["w_dim_2"])

else:
  params["word_file"] = "./words/{}_words.txt".format(data_set)
  params["char_file"] = "./chars/{}_chars.txt".format(data_set)
  params["vector_file"] = "./vectors/{}_{}_{}d.npy".format(vectors, data_set,
    params["w_dim"])
