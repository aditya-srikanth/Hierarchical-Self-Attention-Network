"""
Implements k-fold cross validation
"""
from __future__ import division
import sys
from data_iterator import DataIterator
from data_processor import ATEDataProcessor
from lstm_network import LSTMNetwork
from cnn_network import CNNNetwork
from sklearn.model_selection import train_test_split as split


def get_count(array2D):
  """
  Returns the total number of elements in a list of lists. This method is used to
  get the starting pos id of test ATEDataProcessor object.
  Args:
    array2D (list of lists): list of lists whose total count is to be
    calculated.

  Returns:
    count (int) : total number of elements in the array2D
  """
  count = sum([len(array) for array in array2D])
  return count


def kfold_validate(model, k, kwargs):
  """
  This functin does something similar to k fold validation. We train and test 
  our model k times, by randomly splitting our entire data set into three parts
  (train, dev and test) and return the average of the K runs.
  Args:
      model (str): What kind of model to use. It can be either lstm or cnn
      k (int): Number of iterations over which to average
      kwargs (dict): The parameters that define the model
  
  Returns:
      dict: A dictionary of results, contating the keys precision, recall and 
        fscore.
  """
  p_1 = 0.0
  r_1 = 0.0
  f_1 = 0.0
  train_data = ATEDataProcessor(kwargs["train_file"], **kwargs)
  test_data = ATEDataProcessor(kwargs["test_file"],
                               pos_id=get_count(train_data.annotated_sentences),
                               **kwargs)
  sentences = train_data.annotated_sentences + test_data.annotated_sentences
  for i in range(k):
    print("Run number: {}".format(i))
    train_set, test_set = split(sentences, test_size=0.2, random_state=42)
    train_set, dev_set = split(train_set, test_size=kwargs["test_size"], 
                               random_state=42)
    train = DataIterator(train_set, **kwargs)
    dev = DataIterator(dev_set, **kwargs)
    test = DataIterator(test_set, **kwargs)
    if model == "lstm":
      model = LSTMNetwork(**kwargs)
    elif model == "cnn":
      model = CNNNetwork(max_sentence_length=train_data.max_sentence_len,
                         **kwargs)
    model.build()
    model.train(train, dev)
    results = model.evaluate(test)
    p_1 += float(results["p_1"])
    r_1 += float(results["r_1"])
    f_1 += float(results["f_1"])
    model.close_session()
  print("p_1: {}\nr_1: {}\nf_1: {}".format(p_1/k, r_1/k, f_1/k))
  return {
    "precision": p_1/k,
    "recall": r_1/k,
    "fscore": f_1/k
  }

if __name__ == '__main__':
  if sys.argv[1] == "lstm":
    from params import params
  elif sys.argv[1] == "cnn":
    from cnn_params import params
  kfold_validate(sys.argv[1], k=5, kwargs=params)

