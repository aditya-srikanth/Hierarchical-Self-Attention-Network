"""
Script to train the network
"""
import sys
from data_iterator import DataIterator
from data_processor import ATEDataProcessor
from cnn_network import CNNNetwork
from lstm_network import LSTMNetwork


def get_count(array2D):
  count = sum([len(array) for array in array2D])
  return count


def main(network_type):
  if network_type == "cnn":
    print("Testing CNN network")
    from cnn_params import params
  if network_type == "lstm":
    print("Testing LSTM network")
    from params import params
  train_data = ATEDataProcessor(params["train_file"], **params)
  test_data = ATEDataProcessor(params["test_file"],
                               pos_id=get_count(train_data.annotated_sentences),
                               **params)

  test_set = test_data.annotated_sentences
  test = DataIterator(test_set, word_file=params["word_file"],
                      char_file=params["char_file"])
  if network_type == "cnn":
    model = CNNNetwork(max_sentence_length=train_data.max_sentence_len,
                       **params)
  elif network_type == "lstm":
    model = LSTMNetwork(**params)
  model.build()
  model.restore_session(model.model_directory)
  model.evaluate(test)

if __name__ == "__main__":
  main(sys.argv[1])

