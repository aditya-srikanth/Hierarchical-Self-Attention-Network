data_set = "rest"
params = {
  "dependencies": "./dependencies/{}_dependencies.npy".format(data_set),
  "dependencies_pos": "./dependencies/{}_dependencies_pos.npy".format(data_set),
  "dependencies_ner": "./dependencies/{}_dependencies_ner.npy".format(data_set),
  "pos_file": "./pos_npy/{}_pos.npy".format(data_set),
  "pos_train_file": "./pos_files/{}_train_pos.txt".format(data_set),
  "pos_test_file":  "./pos_files/{}_test_pos.txt".format(data_set),
  "train_file": "./train_data/{}_train.xml".format(data_set),
  "test_file": "./test_data/{}_test.xml".format(data_set),
  "use_pos": True
}
