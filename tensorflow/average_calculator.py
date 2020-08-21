"""
Implements k-fold cross validation
"""
from __future__ import division
import sys
from data_iterator import DataIterator
from data_processor import ATEDataProcessor
from data_generator import generate_data
from lstm_network import LSTMNetwork
from cnn_network import CNNNetwork
from sklearn.model_selection import train_test_split as split
from confidence_interval import mean_confidence_interval

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


def average_calculator(model, k, kwargs, gen_data=True):
  if gen_data:
    generate_data(kwargs) 
  p_1 = 0.0
  r_1 = 0.0
  f_1 = 0.0
  f_scores = []
  train_data = ATEDataProcessor(kwargs["train_file"], **kwargs)
  test_data = ATEDataProcessor(kwargs["test_file"],
                               pos_id=get_count(train_data.annotated_sentences),
                               **kwargs)
  for i in range(k):
    print("Run number: {}".format(i))
    test_set = test_data.annotated_sentences
    train_set, dev_set = split(train_data.annotated_sentences,
                               test_size = kwargs["test_size"])
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
    model.restore_session(model.model_directory)
    results = model.evaluate(test)
    f_scores.append(results["f_1"])
    p_1 += float(results["p_1"])
    r_1 += float(results["r_1"])
    f_1 += float(results["f_1"])
    model.close_session()
  print("p_1: {}\nr_1: {}\nf_1: {}".format(p_1/k, r_1/k, f_1/k))
  print(mean_confidence_interval(f_scores))
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
  average_calculator(sys.argv[1], k=10, kwargs=params)

