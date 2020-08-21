## Prerequisites

nltk, tensorflow, numpy, pycrfsuite, stanford-corenlp

## Installing

Just clone the repository and install the above packages using pip. The code works for python 2

## Running the tests

Any change to the hyperparameters or arguments, make those changes in params.py file. 
* To generate dataset, run python data_processor.py
* To train the model, python train.py
* To test the model, python test.py 

## Getting different RNN models:

To train and test an RNN network, do make lstm_run. The lstm_network.py has the code for the RNN networks, and params.py defines the
hyper parameters for the RNN network. In params.py,
* keep use_pos = False , if you don't want to use pos embeddings
* enter data_set = "rest" to deal with restaurant dataset and data_set = "lap" to deal with laptop dataset.
* the two variables `vectors` and `vectors2` define which pre-trained vectors to use. `vectors` defines the word embeddings used in the
  primary RNN layer (in two/three layer RNN, the second layer; in one layer RNN, the first layer). In three layer RNN, 
  `vectors2` defines word embeddings for the third bi-lstm layer. The variable params["w_dim"] defines the dimensionality of word vectors. 
  Using both of those values, the file name containing the word embeddings is calculated as {vectors}_{params["w_dim"]}d.txt. Therefore the
  values of `vectors` and `vectors2` depends on the name of the file containing your word embeddings. I've "glove_100d.txt", "glove_300d.txt",
  "fasttext_300d.txt", "word2vec_300d.txt", "amazon_300d.txt", "fasttext_skip_rest_100d.txt", "fasttext_skip_lap_100d.txt", "fasttext_cbow_rest_100d.txt",
  "fasttext_cbow_rest_100d.txt". If you want to use your own vectors, then name the file in the above format, and place it in the same directory as params.py
  file. In vector file, each line should be of the form "word word_vector". All terms seperated by white-space.
* If params["crf"] = True, the network will use CRF, otherwise it won't.
* If params["use_char_embeddings"] = True, the network will have additional RNN layer to get embedding of a word from characters of a given word. If this is kept true,
  then params["c_dim"] will give the dimension of character based embeddings.
* If params["bidirectional"] = True, the network will have bi-directional RNNs instead of uni-directional RNNs.
* If params["multi_rnn"] = True, the network will have an additional RNN that takes as input the concatenation of the output from the above mentioned primary RNN layer and 
  the word embeddings identified by "vectors2" variable. NOTE: this only works if params["bidirectional"] = True as well. Will fix sometime soon.
* The default RNN used is LSTM. If params["use_rnn"] = True, a basic Jordan RNN is used. If params["use_gru"] = True, we use GRU instead.
* If use_pos = True, we use parts of speech too in forms of pos embeddings (one hot vectors). That can be accompolished in three ways:
    - concatenate the pos embeddings with inputs to the primary RNN layer. This is accompolished by using params["concat_pos"] = True
    - Attach the pos embeddings to the output of the final RNN layer and pass it to an ANN. To use this, make params["pos_ann"] = True, and params["concat_pos"] = False.
      (If you put params["concat_pos"] = True, the pos will be concatenated to the input to the primary RNN layer, but we will use ANN instead of dense layer at the end)
    - Use a bi-partite ANN at the output layer. For this put params["concat_pos"] = False, and params["bi_ann"] = True
* If params["use_window"] = True, then we use a windowsize of 3 (one to the left, one to the right). We concatenate word vectors of all the words in the windowsize and use that as
  input for the central word. The window size is hardcoded to 3. 
* params["test_size"] is a value between 0 and 1 that defines how much proportion of the training data should be used as validation data
* params["lr"] and params["lr_decay"] are learning reate and learning rate decay respectively.
* If params["small_pos"] is True, we use 6 dimensional one hot pos embeddings instead of the 36

## Getting different CNN models:

To train and test CNN network, simply do make cnn_run. The network is described in cnn_network.py and the hyper parameters are defined in cnn_params.py.
In cnn_params.py,
* The data_set, use_pos, vectors, params["crf"], params["w_dim"] serve the same purpose as lstm_params.py
* params["pos_ann"], params["concat_pos"], params["single_ann"], params["bi_ann"] serve the same purpose as the above. Except here if params["concat_pos"] is True, the pos embeddings are
  concatenated with output of second CNN.
* The truly new thing here is params["hybrid"]. If params["hybrid"] = True, we stack a bi-directional RNN over the CNN. This RNN is by default LSTM. If params["use_rnn"] is True, we use Jordan RNN
  and if params["use_gru"] = True, the RNN is GRU. Keep only one of them as True.