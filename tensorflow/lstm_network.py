"""
This module defines the b-lstm and CRF architecture to be used for ATE.
"""
from __future__ import division
import os
import numpy as np
import tensorflow as tf

from copy import deepcopy
from data_iterator import minbatch_generator
from vocab_builder import get_ids, load_vocab_file, get_aspect_chunks

try:
  from elman_rnn import ElmanRNNCell
except ImportError as e:
  print "your version of tensorflow doesn't support elman implementation"
  print "use tensorflow version > 1.6.0"

class LSTMNetwork(object):
  """
  A class which defines LSTM Neural network for ATE
  """
  def __init__(self, **kwargs):
    """
    initializer for LSTMNetwork object
    Args:
      **kwargs:
        word_file (str): path to the file containing all words in our vocab
        char_file (str): path to the file containing all the chars in our vocab
        vector_file (str): path to the npy file containing our word vectors.
        w_dim (100, int) : number of dimensions of our embedding
        c_dim (100, int) : number of dimensions in our character embedding
        epochs (15, int) : number of epochs to train our data
        dropout (0.5, float) : to prevent over fitting
        batch_size(20, int) : how many training examples we can train at once
        lr (0.001, float) : learning rate
        nepoch_no_improv (3, int): The number of epochs to wait before
          terminating if there is no improvement of the model's performance on
          dev set.
        lr_decay (0.9, float) : can be used for exponential decay of learning
          rate
        char_hidden_size (100, int) : size of our character embedding layer
        word_hidden_size (100, int) : size of our word embedding layer for
          b-lstms
        pos_ann_count(6, int): When we are using pos and when have concat_pos
          as false, we implement a bipartite ann. The 36 dimensional pos
          vector is one input to the network. The number of nuerons in the
          hidden layer corresponding to this input is pos_ann_count
        lstm_ann_count(30, int): The output word representation of the second
          bi-lstm layer is another input to the above ANN. ltm_ann_count is
          number of nuerons in the hidden layer corresponding to this input.

    """
    self.sess = None
    self.saver = None
    self.w_dim = kwargs.get("w_dim", 300)
    self.c_dim = kwargs.get("c_dim", 100)
    self.epochs = kwargs.get("epochs", 15)
    self.dropout = kwargs.get("droput", 0.5)
    self.batch_size = kwargs.get("batch_size", 20)
    self.learning_rate = kwargs.get("lr", 0.001)
    self.lr_decay = kwargs.get("lr_decay", 0.9)
    self.hidden_char_size = self.c_dim
    self.hidden_lstm_size = self.w_dim
    self.nepoch_no_improv = kwargs.get("nepoch_no_improv", 3)
    self.word_vocab = kwargs.get("word_file")
    self.words = load_vocab_file(self.word_vocab)
    self.tags = {"O": 0, "B": 1, "I": 2}
    self.tag_count = 3

    self.char_vocab = kwargs.get("char_file")
    self.chars = load_vocab_file(self.char_vocab)
    self.word_count = len(self.words)
    self.char_count = len(self.chars)

    self.word_vectors = kwargs.get("vector_file")
    self.embeddings = np.load(self.word_vectors)
    self.train_op = None
    self.loss = None
    self.output_directory = "lstm_results/"
    self.model_directory = "lstm_results/model_weights/"

    self.merged = None
    self.file_writer = None

    self.word_ids = None
    self.sequence_lengths = None
    self.char_ids = None
    self.word_lengths = None
    self.labels = None
    self.lr = None
    self.dropout_holder = None
    self.word_embeddings = None
    self.logits = None
    self.labels_pred = None

    self.crf = kwargs.get("crf", False)
    self.trans_params = None
    self.single_ann = kwargs.get("single_ann", False)

    self.pos = kwargs.get("use_pos", False)
    self.concat_pos = kwargs.get("concat_pos", False)
    self.single_ann = kwargs.get("single_ann", False)
    self.bi_ann = kwargs.get("bi_ann", False)
    self.pos_ann = kwargs.get("pos_ann", False)
    self.lstm_ann_count = kwargs.get("lstm_ann_count", 10) 
    self.pos_ann_count = kwargs.get("pos_ann_count", 20)
    self.pos_ids = None
    self.pos_vecs = None
    if self.pos:
      self.pos_embeddings = np.load(kwargs.get("pos_file"))

    self.bidirectional = kwargs.get("bidirectional", True)
    self.gru = kwargs.get("use_gru", False)
    self.rnn = kwargs.get("use_rnn", False)
    self.elman = kwargs.get("use_elman", False)
    self.use_char_embeddings = kwargs.get("use_char_embeddings", True)
    self.use_window = kwargs.get("use_window", False)

    self.multi_rnn = kwargs.get("multi_rnn", False)
    if self.multi_rnn:
      print("using {} for second bi rnn".format(kwargs.get("vector_file_2")))
      self.embeddings_temp = np.load(kwargs.get("vector_file_2")) 

  def add_train_op(self, loss):
    """
    Add a training operator to the model.
    Args:
      loss (tf.float32 tensor): The tensor to be minimized
    Returns:
      None
    """
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.minimize(loss)

  def add_word_embedding_op(self):
    """
    Adds an operation for both word and character embeddings. For word
    embeddings we'll just use our pretrained vectors. For character embeddings
    we'll have to use a blstm layer.
    We also add pos embeddings here. In case we need any other embeddings to
    add, add here. Use embedding layers to avoid overfilling your memory.
    Returns:
      None
    """
    # for POS
    if self.pos:
      print("adding pos embeddings")
      with tf.variable_scope("pos"):
        _pos_embeddings = tf.Variable(self.pos_embeddings,
                                      name="la_pos_embeddings",
                                      dtype=tf.float32, trainable=False)
        pos_embeddings = tf.nn.embedding_lookup(_pos_embeddings, self.pos_ids,
                                                name="pos_embeddings")
        
        self.pos_vecs = pos_embeddings
    # for words
    print("adding word embeddings")
    if self.multi_rnn:
      print("loading embeddings for mult_rnn")
      with tf.variable_scope("word_temp"):
        _word_embeddings_t = tf.Variable(self.embeddings_temp, name="_word_embeddings_t",
                                         dtype=tf.float32, trainable=False)
        word_embeddings_temp = tf.nn.embedding_lookup(_word_embeddings_t,
                                                      self.word_ids,
                                                      name="word_embeddings")
        self.word_embeddings_temp = word_embeddings_temp
    with tf.variable_scope("words"):
      _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings",
                                     dtype=tf.float32, trainable=False)
      word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                               self.word_ids,
                                               name="word_embeddings")
      if self.use_window:
        print("Concatenating word vectors of context words")
        word_embeddings_sl = tf.nn.embedding_lookup(_word_embeddings,
                                                    self.word_ids_sl,
                                                    name="word_embeddings_sl")
        word_embeddings_sr = tf.nn.embedding_lookup(_word_embeddings,
                                                    self.word_ids_sr,
                                                    name="word_embeddings_sr")
        word_embeddings = tf.concat([word_embeddings_sr, word_embeddings,
                                     word_embeddings_sl], axis=-1)
    # for chars
    if self.use_char_embeddings:
      print("adding character embeddings")
      with tf.variable_scope("chars"):
        # randomly intiialize character vectors.
        _char_embeddings = tf.get_variable(name="_char_embeddings",
                                           dtype=tf.float32,
                                           shape=[self.char_count, self.c_dim])
        char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                 self.char_ids,
                                                 name="char_embeddings")
        # reshape the char_embeddings so that we can use time dimension for blstm
        s = tf.shape(char_embeddings)
      # reshape char_embeddings
        char_embeddings = tf.reshape(char_embeddings, shape=[s[0]*s[1], s[-2],
                                                           self.c_dim])
        word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

      # we need two cells for a bidirectional lstm.
      
        if self.gru:
          print("Using GRU for Char embeddings")
          cell_fw = tf.contrib.rnn.GRUCell(self.hidden_char_size)
          cell_bw = tf.contrib.rnn.GRUCell(self.hidden_char_size)
          _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                    char_embeddings,
                                                    sequence_length=word_lengths,
                                                    dtype=tf.float32)
          _, (output_fw, output_bw) = _output
        elif self.rnn:
          print("Using RNN for Char embeddings")
          cell_fw = tf.contrib.rnn.BasicRNNCell(self.hidden_char_size)
          cell_bw = tf.contrib.rnn.BasicRNNCell(self.hidden_char_size)
          _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                    char_embeddings, 
                                                    sequence_length=word_lengths,
                                                    dtype=tf.float32)
          _, (output_fw, output_bw) = _output
        elif self.elman:
          print("Using ElmanRNN for Char embeddings")
          cell_fw = ElmanRNNCell(self.hidden_char_size)
          cell_bw = ElmanRNNCell(self.hidden_char_size)
          _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                    char_embeddings, 
                                                    sequence_length=word_lengths,
                                                    dtype=tf.float32)
          _, (output_fw, output_bw) = _output
        else:
          print("Using LSTM for Char embeddings")
          cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_char_size, 
                                            state_is_tuple=True)
          cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_char_size, 
                                            state_is_tuple=True)
          _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                    char_embeddings,
                                                    sequence_length=word_lengths,
                                                    dtype=tf.float32)
          _, ((_, output_fw), (_, output_bw)) = _output

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, shape=[s[0], s[1], 2*self.hidden_char_size])
        word_embeddings = tf.concat([word_embeddings, output], axis=-1)
      if self.pos and self.concat_pos:
        print("Concatenting pos embeddings to word embeddings")
        word_embeddings = tf.concat([word_embeddings, pos_embeddings],
                                    axis=-1)
    self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_holder)

  def add_logits_op(self):
    """
    the final layer. this corresponds to scores of each tag.
    We first add a stand bi-lstm layer to get representation of words. Then
    after that we add an ANN to map the high dimensional word representation
    to a number of nuerons equal to our tags. This ANN can be a simple normal
    ANN if we are not using.

    But if we are using pos, and cocat pos is false, then we construct two
    anns. one for lstm word representation and one for pos embeddings,
    then we join the two networks, and have a fully connected network from
    this combined layer with final layer having nuerons equal to number of pos
    tags.
    Returns:
      None
    """
    with tf.variable_scope("bi-lstm"):
      # this will be the second layer of b-lstm.
      if self.gru:
        print("Using GRU for final layer")
        rnn_cell = tf.contrib.rnn.GRUCell
      elif self.rnn:
        print("Using RNN for final layer")
        rnn_cell = tf.contrib.rnn.BasicRNNCell
      elif self.elman:
        rnn_cell = ElmanRNNCell
      else:
        print("Using LSTM for final layer")
        rnn_cell = tf.contrib.rnn.LSTMCell

      if self.bidirectional:
        print("Using bidirectional RNN..")
        cell_fw = rnn_cell(self.hidden_lstm_size)
        cell_bw = rnn_cell(self.hidden_lstm_size)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw, cell_bw, self.word_embeddings,
          sequence_length=self.sequence_lengths, dtype=tf.float32)

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.nn.dropout(output, self.dropout_holder)
        output_reshape = 2*self.hidden_lstm_size
      else:
        print("Using unidirectional RNN..")
        cell = rnn_cell(self.hidden_lstm_size)
        output, _ = tf.nn.dynamic_rnn(cell, self.word_embeddings, 
          sequence_length=self.sequence_lengths, dtype=tf.float32)
        self.x = output = tf.nn.dropout(output, self.dropout_holder)
        output_reshape = self.hidden_lstm_size

    if self.multi_rnn and self.bidirectional: 
      with tf.variable_scope("bi-lstm-2"):
        ncell_fw = rnn_cell(self.hidden_lstm_size)
        ncell_bw = rnn_cell(self.hidden_lstm_size)
        output = tf.concat([self.word_embeddings_temp, output], axis=-1)
        (noutput_fw, noutput_bw), _ = tf.nn.bidirectional_dynamic_rnn(
          ncell_fw, ncell_bw, output, sequence_length=self.sequence_lengths,
          dtype = tf.float32)
        output = tf.concat([noutput_fw, noutput_bw], axis=-1)

    with tf.variable_scope("final_layer"):
      output_reshape_2 = self.tag_count
      pred_pos = None

      if self.pos and not self.concat_pos and (self.single_ann or self.pos_ann):
        # This block concatenates pos vector to lstm output
        print("Concatenating pos to lstm output")
        output = tf.concat([output, self.pos_vecs], axis=-1)
        output_reshape = output_reshape + 36
        if self.single_ann:
          # if it's a dense layer without any hidden network, we shouldn't do 
          # this
          print("Setting up for a single ann!")
          output_reshape_2 = self.lstm_ann_count

      if self.pos and not self.concat_pos and self.bi_ann:
        # this sets up the hidden layer for pos vectors
        print("Setting up network for pos in bi ann")
        output_reshape_2 = self.lstm_ann_count
        w_pos = tf.get_variable("w_pos", dtype=tf.float32,
                                shape=[36, self.pos_ann_count])
        b_pos = tf.get_variable("b_pos", dtype=tf.float32, 
                                shape=[self.pos_ann_count],
                                initializer=tf.zeros_initializer())
        output_pos = tf.reshape(self.pos_vecs, [-1, 36])
        pred_pos = tf.matmul(output_pos, w_pos) + b_pos

      nsteps = tf.shape(output)[1]
      output = tf.reshape(output, [-1, output_reshape])
      w1 = tf.get_variable("w1", dtype=tf.float32,
                           shape=[output_reshape, output_reshape_2])
      b1 = tf.get_variable("b1", shape=[output_reshape_2], dtype=tf.float32,
                           initializer=tf.zeros_initializer())
      pred = tf.matmul(output, w1) + b1
      
      if self.pos_ann:
        print("adding only a dense layer!")

      if self.pos and not self.concat_pos and self.bi_ann:
        # in case of bi partite ann, we have to concatenate output of pos to 
        # this.
        print("Setting up network for bi_ann")
        pred = tf.concat([pred, pred_pos], axis=-1)
        output_reshape_2 = output_reshape_2 + self.pos_ann_count

      if self.pos and not self.concat_pos and (self.single_ann or self.bi_ann):
        print("Adding final layer of ANN")
        w2 = tf.get_variable("w2", dtype=tf.float32, shape=[output_reshape_2, 
                                                            self.tag_count])
        b2 = tf.get_variable("b2", dtype=tf.float32, shape=[self.tag_count],
                             initializer=tf.zeros_initializer())
        pred = tf.matmul(pred, w2) + b2
      self.logits = tf.reshape(pred, [-1, nsteps, self.tag_count])

  def add_pred_op(self):
    """
    to predict. This will be used when we aren't using CRF. we just do the
    argmax of our self.logits output and give that as prediction.
    Returns:
      None.
    """
    if not self.crf:
      self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

  def add_loss_op(self):
    """
    Add loss operation
    """
    if self.crf:
      log_likelihood, trans_prams = tf.contrib.crf.crf_log_likelihood(
        self.logits, self.labels, self.sequence_lengths)
      self.trans_params = trans_prams
      self.loss = tf.reduce_mean(-log_likelihood)

    else:
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=self.labels)

      mask = tf.sequence_mask(self.sequence_lengths)
      losses = tf.boolean_mask(losses, mask)

      self.loss = tf.reduce_mean(losses)

    tf.summary.scalar("loss", self.loss)

  def session_init(self):
    """
    Starts the session and initializes the saver.
    Returns:
      None
    """
    print("starting the session")
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()

  def restore_session(self, weights_directory):
    """
    Reload a session with given weights
    Args:
      weights_directory (str): The path to the directory where weights are present
    Returns:
      None
    """
    print("restoring session from {}".format(weights_directory))
    self.saver.restore(self.sess, weights_directory)

  def save_session(self):
    """
    saves the session weights
    Returns:
      None
    """
    print("saving the session")
    if not os.path.exists(self.model_directory):
      os.makedirs(self.model_directory)
    self.saver.save(self.sess, self.model_directory)

  def close_session(self):
    """
    Closes the tensorflow session.
    Returns:
      None
    """
    self.sess.close()

  def add_summary(self):
    """
    Add tf.summary. useful for tensorboard stuff.
    Returns:
      None
    """
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.output_directory,
                                             self.sess.graph)

  def train(self, train, dev):
    """
    The function which performs training of the network. This can perform
    early stopping and exponential decay of learning rate. Uses adam optmizer.
    Args:
      train (DataIterator): A DataIterator object for training set
      dev (DataIterator): A DataIterator object for dev set

    Returns:
      None
    """
    best_score = 0
    nepoch_no_imporov = 0
    self.add_summary()
    best_epoch = 0
    self.get_total_parameters()
    for epoch in range(self.epochs):
      print("Epoch {:} out of {:}".format(epoch+1, self.epochs))
      scores = self.run_epoch(train, dev, epoch)
      f_score = scores[2]

      if f_score >= best_score:
        nepoch_no_imporov = 0
        self.save_session()
        best_score = f_score
        best_epoch = epoch + 1
      else:
        nepoch_no_imporov += 1
        if nepoch_no_imporov >= self.nepoch_no_improv:
          print("early stopping at epoch {} with no "
                       "imporvement".format(best_epoch+1))
          break

  def evaluate(self, test):
    """
    A helper function to display results after running run_evaluate
    Args:
      test (DataIterator): DataIterator object over test set
    Returns:
      metrics (dict): Dictionary containing precision, recall, fscore values.
    """
    print("Testing model over test set")
    metrics = self.run_evaluate_compound(test)
    msg = " - ".join(["{} {}".format(k, v) for k, v in metrics.items()])
    print("\n")
    print(msg)
    return metrics

  def add_placeholders(self):
    """
    Add entries to the computation graph.
    Returns:
      None
    """
    # word_ids are the ids we obtain from word_vocab dictionary. Max size of
    # this wold be max length of sentence in the given batch. Therefore shape
    # ends up being (batch size, max length of sentence in the batch)
    self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                   name="word_ids")
    self.word_ids_sl = tf.placeholder(tf.int32, shape=[None, None],
                                   name="word_ids_sl")
    self.word_ids_sr = tf.placeholder(tf.int32, shape=[None, None],
                                   name="word_ids_sr")
    # pos_id will be indices to the numpy array which hold pos vecs in one hot
    # representation form. Shape will be (batch size, max length of sentence
    # in the batch)
    self.pos_ids = tf.placeholder(tf.int32, shape=[None, None], name="pos_ids")
    # the shape is the same as batch size.
    self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                           name="sequence_lengths")
    # for a given sentence, we want char ids of all the chars in the sentence.
    #  so the shape ends up being (max length of sentence, max length of
    # word), since we are using batches, the size will end up being (
    # batch_size, max_length of sentence, max length of word)
    self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                   name="char_ids")
    # shape = (batch_size, max length of sentence)
    self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_lengths")
    # shape = (batch_size, max length of sentence in batch)
    self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
    self.dropout_holder = tf.placeholder(dtype=tf.float32, shape=[],
                                         name="dropout")
    self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

  def get_feed_dict(self, words, pos=None, labels=None, lr=None, dropout=None):
    """
    Given some data, this method pads the data and builds a feed dictionary.
    Args:
      words (list of lists): It's a list of sentences.
        eg: it's of the form [sentence, sentence, setence..]
            where, each sentence is [(char_ids,char_ids,char_ids..), (word,
            word, word..)]
            where each char_ids is [char_id, char_id..]
      labels (list): list of ids of the tags corresponding to each word.
      pos (list): list of pos ids of words
      lr (float): Learning rate
      dropout (float): Droput probability

    Returns:
      feed_dict (dict):
        {placeholder: value}

    """
    char_ids, word_ids = zip(*words)

    # now word_ids is a list of lists. each list represents a sentence in
    # terms of word _ids.
    # pad each sentence so that all sentences are of same length.
    sequence_lengths = [len(sentence) for sentence in word_ids]

    word_ids = tf.keras.preprocessing.sequence.pad_sequences(
      word_ids, maxlen=max(sequence_lengths), padding="post", value=0,
      dtype="int32").tolist()

    # now char_ids is a tuple of tuples, where each inner tuple is a tuple of
    # lists. Each inner tuple represents a sentence, each list inside the
    # tuple is a list of ids of characters that make up the word.
    # now we have to pad char_ids in such a way that all words (list of ids)
    # are of same length, and all sentences (inner tuple) are of same lengths.
    max_sentence_length = max(sequence_lengths)
    word_lengths = [[len(word) for word in words_] for words_ in char_ids]

    max_word_length = max([max(map(lambda x: len(x), seq)) for seq in char_ids])

    word_lengths = tf.keras.preprocessing.sequence.pad_sequences(
      word_lengths, maxlen=max_sentence_length, padding="post", value=0,
      dtype="int32").tolist()

    char_ids_padded = []

    for sentence in char_ids:
      padded_word = tf.keras.preprocessing.sequence.pad_sequences(
        sentence, maxlen=max_word_length, padding="post", value=0, dtype="int32"
      ).tolist()

      pad_list = [0] * max_word_length
      padded_word.extend([pad_list for _ in range(max_sentence_length - len(
        padded_word))])

      char_ids_padded.append(padded_word)

    char_ids = char_ids_padded
    if pos:
      # now pad pos list
      pos = tf.keras.preprocessing.sequence.pad_sequences(
        pos, maxlen=max_sentence_length, padding="post", value=0, dtype="int32"
      ).tolist()

    # build feed dictionary now.
    feed = {
      self.word_ids: word_ids,
      self.sequence_lengths: sequence_lengths,
      self.char_ids: char_ids,
      self.word_lengths: word_lengths,
    }

    if self.use_window:
      word_ids_sl = deepcopy(word_ids)
      word_ids_sr = deepcopy(word_ids)

      for row in word_ids_sl:
        row.pop(0)
        row.append(0)

      for row in word_ids_sr:
        row.insert(0, 0)
        row.pop()

      feed[self.word_ids_sl] = word_ids_sl
      feed[self.word_ids_sr] = word_ids_sr

    if self.pos and (pos is not None):
      feed[self.pos_ids] = pos

    if labels is not None:
      label_lengths = [len(label) for label in labels]
      labels = tf.keras.preprocessing.sequence.pad_sequences(
        labels, maxlen=max(label_lengths), padding="post"
      ).tolist()
      feed[self.labels] = labels

    if lr is not None:
      feed[self.lr] = lr

    if dropout is not None:
      feed[self.dropout_holder] = dropout

    return feed, sequence_lengths

  def build(self):
    """
    Calls various methods to build the network
    Returns:
      None
    """
    tf.reset_default_graph()
    self.add_placeholders()
    self.add_word_embedding_op()
    self.add_logits_op()
    self.add_pred_op()
    self.add_loss_op()
    self.add_train_op(self.loss)
    self.session_init()

  def predict_batch(self, words, pos=None):
    """
    Predicts the tags of words for given batch of sentences
    Args:
      words (tensor): tensor representing a batch of sentences
      pos (list) : list containing pos_ids of the setence
    Returns:
      labels_pred (tensor) : list of labels for each sentence in the batch
      sequence_length (list) : length of each sentence in the batch
    """
    feed_dict, sequence_lengths = self.get_feed_dict(words, pos=pos, dropout=1.0)

    if self.crf:
      viterbi_sequences = []
      logits, trans_params = self.sess.run([self.logits, self.trans_params],
                                           feed_dict=feed_dict)
      for logit, sequence_length in zip(logits, sequence_lengths):
        logit = logit[:sequence_length]
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit,
                                                                  trans_params)
        viterbi_sequences += [viterbi_seq]

      return viterbi_sequences, sequence_lengths

    else:
      labels_pred = self.sess.run(self.labels_pred, feed_dict=feed_dict)
      return labels_pred, sequence_lengths

  def run_epoch(self, train, dev, epoch):
    """
    Performs one epoch. That is train over train set and evaluate over dev set.
    Args:
      train (DataIterator object): A DataIterator object over train file
      dev (DataIterator object): A DatatIterator ojbect over dev set
      epoch (int): the index of epoch we are currently running

    Returns:
      tuple (float, float, float):
        precision : over dev set
        recall : over dev set
        fscore : over dev set
    """
    batch_size = self.batch_size
    nbatches = (len(train) + batch_size - 1)//batch_size
    prog = tf.keras.utils.Progbar(target=nbatches)

    for i, (words, labels, pos) in enumerate(minbatch_generator(train,
                                                                batch_size)):
      feed_dict, _ = self.get_feed_dict(words, pos=pos, labels=labels, 
                                        lr=self.learning_rate,
                                        dropout=self.dropout)
      _, train_loss, summary = self.sess.run([self.train_op, self.loss,
                                              self.merged], feed_dict=feed_dict)
      prog.update(i+1, [("train loss", train_loss)])

      if i % 10 == 0:
        self.file_writer.add_summary(summary, epoch*nbatches+i)

    metrics = self.run_evaluate_compound(dev)
    msg = " - ".join(["{} {}".format(k, v) for k, v in metrics.items()])
    print("\n")
    print(msg)
    return metrics["p_1"], metrics["r_1"], metrics["f_1"]

  def run_evaluate(self, data):
    """
    Evaluate performance of our model on a data set
    Args:
      data (DataIterator): Test data set

    Returns:
      metrics (dict) : has precision, recall, fscore for all tags
    """
    print "called run_evaluate"
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for words, labels, pos in minbatch_generator(data, self.batch_size):
      labels_pred, sequence_lengths = self.predict_batch(words, pos)
      if not type(labels_pred) == list:
        labels_pred = labels_pred.tolist()
      for i in range(len(labels)):
        for j in range(len(labels[i])):
          if labels[i][j] == 1:
            if labels_pred[i][j] == 1:
              tp += 1
            else:
              fn += 1
          else:
            if not labels_pred[i][j] == 1:
              tn += 1
            else:
              fp += 1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fscore = (2*recall*precision)/(recall+precision)
    metrics = dict()
    metrics["p_1"] = precision
    metrics["f_1"] = fscore
    metrics["r_1"] = recall
    return metrics

  def run_evaluate_compound(self, data):
    """
    Evaluate performance of our model on a data set with consecutive B+Is 
    merged together.
    Args:
      data (DataIterator): Test data set

    Returns:
      metrics (dict) : has precision, recall, fscore for all tags
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for words, labels, pos in minbatch_generator(data, self.batch_size):
      labels_pred, _ = self.predict_batch(words, pos)
      if not type(labels_pred) == list:
        labels_pred = labels_pred.tolist()
      for i, label_actual in enumerate(labels):
        # we have to now find all aspect terms. That is, merge consecutive B 
        # with I's until we get an O or B
        a_terms_actual, non_a_terms_actual = get_aspect_chunks(label_actual)
        for a_term in a_terms_actual:
          # say we have [1,2,3] as our a_term. we have to make sure that our 
          # prediction should have tags B, I, I in 1, 2 and 3 indices respectively
          for index in a_term:
            if not label_actual[index] == labels_pred[i][index]:
              fn += 1
              break
          else:
            tp += 1
        for index in non_a_terms_actual:
          if label_actual[index] == labels_pred[i][index]:
            tn += 1
          else:
            fp += 1

    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    fscore = (2*recall*precision)/(recall+precision) if recall+precision else 0
    metrics = dict()
    metrics["p_1"] = precision
    metrics["f_1"] = fscore
    metrics["r_1"] = recall
    return metrics

  def predict(self, word_raw):
    """
    returns the predicted tags for given words
    Args:
      word_raw (str): sentence for which we can predict tags
    Returns:
      predictions (list): list of tags. one tag for each word in the sentence
    """
    words = ["<number>" if word.isdigit() else word.lower()
             for word in word_raw.strip().split()]
    words = [get_ids(word, self.words, self.chars) for word in words]
    words = list(zip(*tuple(words)))
    preds, _ = self.predict_batch([words])
    return preds

  def get_total_parameters(self):
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print("{}\t:{}".format(variable.name, variable_parameters))
        total_parameters += variable_parameters
    print("Total number of parameters to train\t:{}".format(total_parameters))

