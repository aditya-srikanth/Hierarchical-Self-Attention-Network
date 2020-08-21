"""
This module defines CNN architecture for ATE
"""
from __future__ import division
import os
import numpy as np
import tensorflow as tf

from copy import deepcopy
from vocab_builder import load_vocab_file, get_ids, get_aspect_chunks
from data_iterator import minbatch_generator
try:
  from elman_rnn import ElmanRNNCell
except ImportError as e:
  print("your version of tensorflow doesn't support Elman implementation")
  print("install tensorflow version >1.6.0")

class CNNNetwork(object):
  """
  A class which defines CNN for ATE
  """
  def __init__(self, **kwargs):
    """
    __init__ for CNNNetwork object
    Args:
      **kwargs:
        w_dim (int, 300): The dimension of word_embedding to use
        epochos (int, 15): Max number of epochs to train the model for
        droput (float, 0.5): The dropout probability
        batch_size (int, 20): The batch size for training or testing
        lr (float, 0.001): Learning rate
        lr_decay (float, 0.9): By how much to decay learning rate by, each epoch
        nepoch_no_improv(int, 3): If after this many epochs we still haven't
          gotten the best score on dev set, we will terminate the training
        word_file (str): The file containing list of words of our corpus
        char_file (str): The file contaning list of chars of our corpus
        vector_file (str): Numpy file containing word vectors
        use_pos (Boolean): Whether or not use pos_embeddings in any of the ways
          we describe below
        concat_pos (Boolean): If both use_pos and concat_pos are true, we
          concatenate pos embedding with word embeddings before giving them as
          input to the CNN layers
        single_ann (Boolean): If use_pos is true, concat_pos is false, this var
          determines whether or not use single ANN architecture after CNN layer
          (or lstm layer if hybrid = True), instead of one dense layer. If we
          are using single ann, pos is concatenated to the output of cnn (or
          lstm layer and is fed to the ann)
        bi_ann (Boolean): If use_pos is true, concat_pos is fale, this variable
          determines whether or not tuse bipartite architecture after CNN/LSTM
          layer, instead of one dense layer.
        pos_ann (Boolean): If use_pos is true, concat_pos is false, single_ann
          is false, bi_ann is false and pos_ann is true, we concatenate pos vec
          to cnn/lstm layer output and pass it to single dense layer without any
          intermidiate hidden layer
        ann_size (int, 10): The size of hidden layer the output of cnn/lstm is
          connected to (in case of single ann, in case of bi ann, the pos is
          concatenated with the cnn output and the connected to a hidden layer
          of this size)
        pos_ann_size (int, 20): The size of hidden layer the pos embedding is
          connected to in bi_ann
        pos_file (str): The path to the npy file containing pos embeddings.
        filter_size_1 (int, 2): Size of filter on first CNN layer
        filter_size_2 (int, 3): Size of filter on second CNN layer
        out_channel_1 (int, 100): Number of filter maps on first CNN layer
        out_channel_2 (int, 50): Number of filter maps on second CNN layer
        strides_1 (int, 1): Size of strides in first CNN layer
        strides_2 (int, 1): Size of strides in second CNN layer
        pool_size_1 (int, 2): Pooling size on first CNN layer
        pool_size_2 (int, 2): Pooling size on second CNN layer.
        max_sentence_length (int): Maximum sentence length in our corpus we are
          building this model over. This will be used to define shapes for all
          the layers
        hybrid (Boolean, False): If hybrid is True, we will use a B-LSTM layer
          after CNN layer.
        lstm_size (int, 50): If hybrid is True, this gives the size of hidden
          state of bi-lstm layer.
    """
    # Place holders and class variables
    self.sess = None
    self.saver = None
    self.sequence_lengths = None
    self.train_op = None
    self.loss = None
    self.word_lengths = None
    self.merged = None
    self.file_writer = None
    self.labels = None
    self.word_ids = None
    self.lr = None
    self.dropout_holder = None
    self.word_embeddings = None
    self.pos_ids = None
    self.pos_vecs = None
    self.cnn_layer = None
    self.logits = None
    self.labels_pred = None
    self.trans_params = None

    # Dataset parameterse
    self.w_dim = kwargs.get("w_dim", 300)
    self.epochs = kwargs.get("epochs", 15)
    self.dropout = kwargs.get("droput", 0.5)
    self.batch_size = kwargs.get("batch_size", 20)
    self.learning_rate = kwargs.get("lr", 0.001)
    self.lr_decay = kwargs.get("lr_decay", 0.9)
    self.nepoch_no_improv = kwargs.get("nepoch_no_improv", 3)
    self.word_vocab = kwargs.get("word_file")
    self.char_vocab = kwargs.get("char_file")
    self.word_vectors = kwargs.get("vector_file")
    self.word_vectors_2 = kwargs.get("vector_file_2")

    # Load dataset
    self.words = load_vocab_file(self.word_vocab)
    self.tags = {"O": 0, "B": 1, "I": 2}
    self.tag_count = 3
    self.chars = load_vocab_file(self.char_vocab)
    self.word_count = len(self.words)
    self.char_count = len(self.chars)
    self.embeddings = np.load(self.word_vectors)

    self.use_additional = kwargs.get("use_additional_embeddings", False)
    if self.use_additional:
      self.additional_embeddings = np.load(self.word_vectors_2)

    self.output_directory = "cnn_results/"
    self.model_directory = "cnn_results/model_weights/"

    # Non convolution network settings
    self.use_char_embeddings = kwargs.get("use_char_embeddings", False)
    if self.use_char_embeddings:
      self.c_filter_width = kwargs.get("c_filter_width", 2)
      self.c_filter_height = kwargs.get("c_filter_height", 2)
      self.c_pool_size = kwargs.get("c_pool_size", 2)
      self.c_strides = kwargs.get("c_strides", 2)
      self.max_word_length = kwargs.get("max_word_length", 50)
      self.c_dim_input = kwargs.get("c_dim_input", 50)
      self.c_dim_output = kwargs.get("c_dim_output", 10)
    self.pos = kwargs.get("use_pos", False)
    self.concat_pos = kwargs.get("concat_pos", False)
    self.single_ann = kwargs.get("single_ann", False)
    self.bi_ann = kwargs.get("bi_ann", False)
    self.pos_ann = kwargs.get("pos_ann", False)
    self.ann_size = kwargs.get("ann_size", 10)
    self.pos_ann_count = kwargs.get("pos_ann_size", 10)
    if self.pos:
      self.pos_embeddings = np.load(kwargs.get("pos_file"))

    self.use_window = kwargs.get("use_window", False)
    self.crf = kwargs.get("crf", False)
    self.lp = kwargs.get("lp")

    if self.lp:
      self.curr_sentence_id = kwargs.get("sentence_id", 0)
      self.dependencies = np.load(kwargs.get("dependencies"))

    self.hybrid = kwargs.get("hybrid")
    if self.hybrid:
      self.lstm_size = kwargs.get("lstm_size", 50)

    # CNN parameters
    self.filter_size_1 = kwargs.get("filter_size_1", 2)
    self.filter_size_2 = kwargs.get("filter_size_2", 3)
    self.out_channels_1 = kwargs.get("out_channels_1", 100)
    self.out_channels_2 = kwargs.get("out_channels_2", 50)
    self.strides_1 = kwargs.get("strides_1", 1)
    self.strides_2 = kwargs.get("strides_2", 1)
    self.pool_size_1 = kwargs.get("pool_size_1", 2)
    self.pool_size_2 = kwargs.get("pool_size_2", 2)
    self.max_len = kwargs["max_sentence_length"]

    self.pos_embedding_size = kwargs.get("pos_embedding_size", 36)

    self.rnn = kwargs.get("use_rnn", False)
    self.gru = kwargs.get("use_gru", False)
    self.elman = kwargs.get("use_elman", False)
    self.max_pooling = kwargs.get("max_pooling", True)
    self.word_ids_sl = None
    self.word_ids_sr = None
    self.use_window_rnn = kwargs.get("use_window_rnn", False)
    self.cnn_values = None

  def session_init(self):
    print("Starting the session")
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()

  def restore_session(self, weights_directory):
    print("restoring session from {}".format(weights_directory))
    self.saver.restore(self.sess, weights_directory)

  def save_session(self):
    print("Saving the session")
    if not os.path.exists(self.model_directory):
      os.makedirs(self.model_directory)
    self.saver.save(self.sess, self.model_directory)

  def close_session(self):
    self.sess.close()

  def add_summary(self):
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.output_directory,
                                             self.sess.graph)

  def add_place_holders(self):
    self.word_ids = tf.placeholder(tf.int32, shape=[None, self.max_len],
                                   name="word_ids")
    self.char_ids = tf.placeholder(tf.int32, shape =[None, None, None],
                                   name="char_ids")
    # todo: replace self.word_ids, self.word_ids_sl, self.word_ids_sr with a
    # single tensor of shape = [3, None, self.max_len] where 3 is the window
    # size here. Here sl is shifted left and sr is shifted right.
    self.word_ids_sl = tf.placeholder(tf.int32, shape=[None, self.max_len])
    self.word_ids_sr = tf.placeholder(tf.int32, shape=[None, self.max_len])
    self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                           name="sequence_lengths")
    self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_lengths")
    self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
    self.dropout_holder = tf.placeholder(dtype=tf.float32, shape=[],
                                         name="dropout")
    self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
    self.pos_ids = tf.placeholder(tf.int32, shape=[None, self.max_len])
    self.pos_vecs = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
    if self.use_window_rnn:
      s = 3*self.out_channels_2
    else:
      s = self.out_channels_2
    self.cnn_values = tf.placeholder(dtype=tf.float32, shape=[None, self.max_len,
                                                              s])

  def add_word_embedding_op(self):
    """
    Add an embedding layer.
    Returns:
      None
    """
    if self.pos:
      print("adding pos embeddings")
      with tf.variable_scope("pos"):
        _pos_embeddings = tf.Variable(self.pos_embeddings,
                                      name="la_pos_embeddings",
                                      dtype=tf.float32, trainable=False)
        pos_embeddings = tf.nn.embedding_lookup(_pos_embeddings, self.pos_ids,
                                                name="pos_embeddings")
        self.pos_vecs = pos_embeddings
    print("adding word_embeddings")
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
    if self.use_char_embeddings:
      print("adding CNN for char embeddings")
      with tf.variable_scope("chars"):
        _char_embeddings = tf.get_variable(name="_char_embeddings",
                                           dtype=tf.float32,
                                           shape=[self.char_count, 
                                                  self.c_dim_input])
        char_embeddings = tf.nn.embedding_lookup(_char_embeddings, 
                                                 self.char_ids, 
                                                 name="char_embeddings")
        s = char_embeddings.shape
        # the shape of our char_embeddings is now (batch_size, max number of words
        # in each sentence, max number of chars in each word, self.c_dim )
        char_filter = tf.get_variable("char_filter", dtype=tf.float32,
                                      shape=[self.c_filter_width, 
                                             self.c_filter_height,
                                             self.c_dim_input,
                                             self.c_dim_output])
        print("adding 2d convolution layer")
        char_conv_layer = tf.nn.conv2d(char_embeddings, char_filter, 
                                       strides=[1, 1, 1, 1], 
                                       padding="SAME")
        char_conv_layer = tf.nn.tanh(char_conv_layer)
        print("adding 2d pooling layer")
        char_conv_layer = tf.layers.max_pooling2d(char_conv_layer, 
                                                  1, 
                                                  strides=1)
        char_output = tf.reshape(char_conv_layer, shape=[-1, self.max_len, 
                                                         self.max_word_length*
                                                         self.c_dim_output])
        word_embeddings = tf.concat([word_embeddings, char_output], axis=-1)
    if self.pos and self.concat_pos:
      print("concatenating pos with word_embeddings")
      word_embeddings = tf.concat([word_embeddings, pos_embeddings], axis=-1)
    self.word_embeddings = word_embeddings
    if self.use_additional and self.hybrid:
      print("using additional embeddings")
      _word_embeddings_2 = tf.Variable(self.additional_embeddings,
                                       name="two_word_embeddings",
                                       dtype=tf.float32, trainable=False)
      word_embeddings_2 = tf.nn.embedding_lookup(_word_embeddings_2,
                                                  self.word_ids,
                                                  name="two_word_embeddings")
      self.word_embeddings_2 = word_embeddings_2


  def add_convolution_layers(self):
    """
    our model has two convolution layers. Each convolution layer has a
    maxpooling layer. The second convolution layer has dropout applied to it.
    This method adds these two layers.
    Returns:
      None
    """
    with tf.variable_scope("conv_layer_1"):
      # our input to this layer is from word embeddings. which would be of
      # shape (batch_size, number of words in the sentence, dimension of
      # embedding). For us dimension of embedding is the number of input
      # channels, self.out_channels_1 is the output channels.

      # First we have to define a filter. The shape of the filter would be [
      # self.filter_size_1, dimension of embedding, self.out_channels_1]
      filter1 = tf.get_variable("filter1", dtype=tf.float32,
                                shape=[self.filter_size_1,
                                       self.word_embeddings.shape[2],
                                       self.out_channels_1])
      conv_layer_1 = tf.nn.conv1d(self.word_embeddings, filters=filter1,
                                  stride=self.strides_1, padding="SAME")
      conv_layer_1 = tf.nn.tanh(conv_layer_1)
      # now add pooling layer
      print("adding pooling to layer1")

      if self.max_pooling:
        print("Adding max pooling layer")
        pooling_layer = tf.layers.max_pooling1d
      else:
        print("Adding average pooling layer")
        pooling_layer = tf.layers.average_pooling1d
      conv_layer_1 = pooling_layer(conv_layer_1,
                                   pool_size=self.pool_size_1,
                                   strides=self.strides_1,
                                   padding="SAME")

    with tf.variable_scope("conv_layer_2"):
      print("adding convolution layer 2")
      # Input to this layer is conv_layer_1 from above.
      # get another filter now. Shape of this filter would be [
      # self.filter_size_2, self.out_channels_1, self.out_channels_2]
      filter2 = tf.get_variable("filter2", dtype=tf.float32,
                                shape=[self.filter_size_2,
                                       self.out_channels_1,
                                       self.out_channels_2])
      conv_layer_2 = tf.nn.conv1d(conv_layer_1, filters=filter2,
                                  stride=self.strides_2, padding="SAME")
      conv_layer_2 = tf.nn.tanh(conv_layer_2)
      print("adding pooling to layer2")
      conv_layer_2 = pooling_layer(conv_layer_2,
                                   pool_size=self.pool_size_2,
                                   strides=self.strides_2,
                                   padding="SAME")

    self.cnn_layer = tf.nn.dropout(conv_layer_2, self.dropout_holder)
    self.word_reps = self.cnn_layer
    if self.use_additional and self.hybrid:
      self.cnn_layer = tf.concat([self.word_reps, self.word_embeddings_2], 
                                  axis=-1)
      self.word_reps = self.cnn_layer

  def add_dense_layer(self):
    """
    After two convolution layers are added, we need to add a densely
    connected layer for predictions.
    Returns:
      None
    """
    output = self.cnn_layer
    # weight_shape is dimension 1 of the first weights in the dense layer. We
    # update it accordingly as we keep adding intermediate layers
    weight_shape = self.out_channels_2
    if self.hybrid:
      with tf.variable_scope("lstm_layer"):
        print("adding Recurrent layer")
        if self.rnn:
          print("Adding Basic RNN cell")
          cell_fw = tf.contrib.rnn.BasicRNNCell(self.lstm_size)
          cell_bw = tf.contrib.rnn.BasicRNNCell(self.lstm_size)
        elif self.gru:
          print("Adding GRU cell")
          cell_fw = tf.contrib.rnn.GRUCell(self.lstm_size)
          cell_bw = tf.contrib.rnn.GRUCell(self.lstm_size)
        elif self.elman:
          print("Adding ELMAN cell")
          cell_fw = ElmanRNNCell(self.lstm_size)
          cell_bw = ElmanRNNCell(self.lstm_size)
        else:
          print("Adding LSTM cell")
          cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_size)
          cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_size)
        if self.use_window_rnn:
          cnn_values = self.cnn_values
          print("using window_rnn")
        else:
          print("not using window_rnn")
          cnn_values = self.cnn_layer
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw, cell_bw, cnn_values,
          sequence_length=self.sequence_lengths, dtype=tf.float32
        )
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.nn.dropout(output, self.dropout_holder)
        self.word_reps = output
        weight_shape = 2*self.lstm_size
    with tf.variable_scope("dense_layer"):
      print("adding dense layer")
      # weight_shape_2 is dimension 2 of the first weights in the dense layer.
      # we update it accordingly if we add more layers at the end
      weight_shape_2 = self.tag_count
      if self.pos and (self.single_ann or self.pos_ann):
        # this if block is for the cases when we have to concatenate the output
        # from prevoius layer to pos_embeddings
        print("Concatenating pos vecs with lstm/cnn output")
        output = tf.concat([output, self.pos_vecs], axis=-1)
        weight_shape = weight_shape + self.pos_embedding_size
        output = tf.reshape(output, [-1, weight_shape])
        if self.single_ann:
          # This means we will have a hidden layer i.e two weights.
          print("Setting up hidden layer for single ann")
          weight_shape_2 = self.ann_size
      if self.pos and not self.concat_pos and self.bi_ann:
        # this if block si for the case when we want to implement a bi-partite
        # ann, as such we will get weights for pos embeddings
        print("setting up network for bi ann..")
        weight_shape_2 = self.ann_size
        w_pos = tf.get_variable("w_pos", dtype=tf.float32,
                                shape=[self.pos_embedding_size,
                                       self.pos_ann_count])
        b_pos = tf.get_variable("b_pos", dtype=tf.float32,
                                shape=[self.pos_ann_count],
                                initializer=tf.zeros_initializer())
        output_pos = tf.reshape(self.pos_vecs, [-1, self.pos_embedding_size])
        pred_pos = tf.matmul(output_pos, w_pos) + b_pos
        output = tf.reshape(output, [-1, weight_shape])
      if len(output.shape) == 3:
        print("Setting up a network with no ann, only a fully connected dense "
              "layer")
        # we have to reshape our output so that we can multiply it with weights.
        # if we reach here, it means we haven't reshaped it properly.
        output = tf.reshape(output, [-1, weight_shape])
      w1 = tf.get_variable("w1", dtype=tf.float32,
                           shape=[weight_shape, weight_shape_2])
      b1 = tf.get_variable("b1", dtype=tf.float32, shape=[weight_shape_2],
                           initializer=tf.zeros_initializer())
      pred = tf.matmul(output, w1) + b1
      # the pred above has weight_shape_2 as it's second dimension. our vars
      # have been defined in such a way that weight_shape_2 = self.tag_count if
      # we don't have to care about adding another layer.
      if self.pos and not self.concat_pos and self.bi_ann:
        # This block deals with concatenating output of pos hidden layer to
        # pred. We change weight_shape_2 accordingly.
        print("setting up network for bi ann..")
        pred = tf.concat([pred, pred_pos], axis=-1)
        weight_shape_2 = weight_shape_2 + self.pos_ann_count

      if self.pos and (self.single_ann or self.bi_ann):
        # now, these are our final sets of weights in case we have an ann. so
        # the second dimension of these weights is self.tag_count!
        print("setting up network for ann")
        w2 = tf.get_variable("w2", dtype=tf.float32, shape=[weight_shape_2,
                                                            self.tag_count])
        b2 = tf.get_variable("b2", shape=[self.tag_count], dtype=tf.float32,
                             initializer=tf.zeros_initializer())
        pred = tf.matmul(pred, w2) + b2
      self.logits = tf.reshape(pred, [-1, self.max_len, self.tag_count])

  def add_pred_op(self):
    if not self.crf:
      self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

  def add_loss_op(self):
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

  def get_feed_dict(self, words, pos=None, labels=None, lr=None, dropout=None):
    char_ids, word_ids = zip(*words)
    sequence_lengths = [len(sentence) for sentence in word_ids]
    word_ids = tf.keras.preprocessing.sequence.pad_sequences(
      word_ids, maxlen=self.max_len, padding="post", value=0,
      dtype="int32").tolist()
    if pos:
      pos = tf.keras.preprocessing.sequence.pad_sequences(
        pos, maxlen=self.max_len, padding="post", value=0, dtype="int32"
      ).tolist()

    word_ids_sl = deepcopy(word_ids)
    word_ids_sr = deepcopy(word_ids)

    for row in word_ids_sl:
      # shift each row by an element to left. so [1, 2, 3] becomes [2, 3, 0].
      row.pop(0)
      row.append(0)

    for row in word_ids_sr:
      # shift each row by an element to right. so [1, 2, 3] becomes [0, 1, 2]
      row.insert(0, 0)
      row.pop()
    # for a window size of three, word_ids_sr, word_ids_srl provide left and
    # right context. for example word left to 1 (0) is word_ids_sr[0] and word
    # right to 1 (2) is word_ids_sl[0]
    feed = {
      self.word_ids: word_ids,
      self.word_ids_sl: word_ids_sl,
      self.word_ids_sr: word_ids_sr,
      self.sequence_lengths: sequence_lengths,
    }
    if self.use_char_embeddings:
      char_ids_padded = []
      for sentence in char_ids:
        padded_word = tf.keras.preprocessing.sequence.pad_sequences(
          sentence, maxlen=self.max_word_length, padding="post", value=0, 
          dtype="int32").tolist()

        pad_list = [0] * self.max_word_length
        padded_word.extend([pad_list for _ in 
                            range(self.max_len-len(padded_word))])
        char_ids_padded.append(padded_word)
      feed[self.char_ids] = char_ids_padded
    if self.pos and (pos is not None):
      feed[self.pos_ids] = pos

    if labels is not None:
      labels = tf.keras.preprocessing.sequence.pad_sequences(
        labels, maxlen=self.max_len, padding="post"
      ).tolist()
      feed[self.labels] = labels

    if lr is not None:
      feed[self.lr] = lr

    if dropout is not None:
      feed[self.dropout_holder] = dropout

    return feed, sequence_lengths

  def add_train_op(self, loss):
    """
    Add a training operation to the model
    Args:
      loss (tf.float32 tensor): tensor to be optimized
    Returns:
      None
    """
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.minimize(loss)

  def build(self):
    tf.reset_default_graph()
    self.add_place_holders()
    self.add_word_embedding_op()
    self.add_convolution_layers()
    self.add_dense_layer()
    self.add_pred_op()
    self.add_loss_op()
    self.add_train_op(self.loss)
    self.session_init()

  def train(self, train, dev):
    best_score = 0
    nepoch_no_improv = 0
    best_epoch = 0
    self.add_summary()
    self.get_total_parameters()
    for epoch in range(self.epochs):
      print("Epoch {:} out of {:}".format(epoch + 1, self.epochs))
      scores = self.run_epoch(train, dev, epoch)
      f_score = scores[2]

      if f_score >= best_score:
        nepoch_no_improv = 0
        self.save_session()
        best_score = f_score
        best_epoch = epoch
      else:
        nepoch_no_improv += 1
        if nepoch_no_improv >= self.nepoch_no_improv:
          print("early stopping at epoch {} with no "
                "imporvement".format(best_epoch))
          break

  def evaluate(self, test):
    print("Testing model over test set")
    metrics = self.run_evaluate_compound(test)
    msg = " - ".join(["{} {}".format(k, v) for k, v in metrics.items()])
    print("\n")
    print(msg)
    return metrics

  def predict_batch(self, words, pos=None):
    feed_dict, sequence_lengths = self.get_feed_dict(words, pos=pos, dropout=1.0)
    if self.use_window_rnn:
      cnn_values = self.sess.run(self.cnn_layer, feed_dict=feed_dict)
      cnn_values_sl = deepcopy(cnn_values)
      cnn_values_sr = deepcopy(cnn_values)
      for j in xrange(cnn_values.shape[0]):
        row = list(cnn_values_sl[j])
        row.pop(0)
        row.append([0]*row[0])
        cnn_values_sl[j] = np.array(row)

        row = list(cnn_values_sr[j])
        row.insert(0, [0]*row[0])
        row.pop()
        cnn_values_sr[j] = np.array(row)
      cnn_values_concat = np.concatenate([cnn_values_sr, cnn_values,
                                   cnn_values_sl], axis=-1)
      feed_dict[self.cnn_values] = cnn_values_concat

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

  def predict_with_lp(self, sentence_id):
    raise NotImplemented("Haven't implemented Linguistic patterns yet.")

  def run_epoch(self, train, dev, epoch):
    batch_size = self.batch_size
    nbatches = (len(train) + batch_size - 1)//batch_size
    prog = tf.keras.utils.Progbar(target=nbatches)

    for i, (words, labels, pos) in enumerate(minbatch_generator(train,
                                                                batch_size)):
      feed_dict, _ = self.get_feed_dict(words, pos=pos, labels=labels,
                                        lr=self.learning_rate,
                                        dropout=self.dropout)
      if self.use_window_rnn:
        cnn_values = self.sess.run(self.cnn_layer, feed_dict=feed_dict)
        cnn_values_sl = deepcopy(cnn_values)
        cnn_values_sr = deepcopy(cnn_values)
        for j in xrange(cnn_values.shape[0]):
          row = list(cnn_values_sl[j])
          row.pop(0)
          row.append([0]*len(row[0]))
          cnn_values_sl[j] = np.array(row)

          row = list(cnn_values_sr[j])
          row.insert(0, [0]*len(row[0]))
          row.pop()
          cnn_values_sr[j] = np.array(row)
        cnn_values_concat = np.concatenate([cnn_values_sr, cnn_values,
                                           cnn_values_sl], axis=-1)
        feed_dict[self.cnn_values] = cnn_values_concat
        _, train_loss = self.sess.run([self.train_op, self.loss],
                                    feed_dict=feed_dict)
      else:
        _, train_loss = self.sess.run([self.train_op, self.loss],
                                      feed_dict=feed_dict)
      prog.update(i+1, [("train loss", train_loss)])

    metrics = self.run_evaluate_compound(dev)
    msg = " - ".join(["{} {}".format(k, v) for k, v in metrics.items()])
    print("\n")
    print(msg)
    return metrics["p_1"], metrics["r_1"], metrics["f_1"]

  def run_evaluate(self, data):
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
            if labels_pred[i][j] == 0:
              tn += 1
            else:
              fp += 1
    if tp+fp == 0:
      precision = 0
    else:
      precision = tp/(tp+fp)
    if tp+fn == 0:
      recall = 0
    else:
      recall = tp/(tp+fn)
    if recall+precision == 0:
      fscore = 0
    else:
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
      labels_pred, sequence_lengths = self.predict_batch(words, pos)
      if not type(labels_pred) == list:
        labels_pred = labels_pred.tolist()
      for i, label_actual in enumerate(labels):
        # we have to now find all aspect terms. That is, merge consecutive B 
        # with I's until we get an O or B
        a_terms_actual, non_a_terms_actual = get_aspect_chunks(label_actual)
        for a_term in a_terms_actual:
          # say we have [1,2,3] as our a_term. we have to make sure that our 
          # prediction should have tags B, I, I in 1, 2 and 3 indices respectively
          a_term_pred = []
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

    precision = 0 if tp+fp == 0 else tp/(tp+fp)
    recall = 0 if tp+fn == 0 else tp/(tp+fn)
    fscore = 0 if precision+recall == 0 else (2*recall*precision)/(recall+precision)
    metrics = dict()
    metrics["p_1"] = precision
    metrics["f_1"] = fscore
    metrics["r_1"] = recall
    return metrics


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

  def get_word_reps(self, words):
    words = ["<number>" if word.isdigit() else word.lower()
             for word in words]
    s_len = len(words)
    words = [get_ids(word, self.words, self.chars) for word in words]
    words = [list(zip(*tuple(words)))]
    feed_dict, _ = self.get_feed_dict(words, dropout=0.5)
    word_reps = self.sess.run(self.word_reps, feed_dict=feed_dict)
    return word_reps[0][:s_len]

