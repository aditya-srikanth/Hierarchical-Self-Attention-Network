import sys
from data_iterator import DataIterator
from data_processor import ATEDataProcessor
from cnn_network import CNNNetwork
from lstm_network import LSTMNetwork
from sklearn.model_selection import train_test_split as split


def main(network_type):
  if network_type == "cnn":
    print("Training CNN network")
    from cnn_params import params
  if network_type == "lstm":
    print("Training LSTM network")
    from params import params
  train_data = ATEDataProcessor(params["train_file"], **params)
  sentences = train_data.annotated_sentences
  train_set, dev_set = split(sentences, test_size=params["test_size"])
  train = DataIterator(train_set, word_file=params["word_file"],
                       char_file=params["char_file"])
  dev = DataIterator(dev_set, word_file=params["word_file"],
                     char_file=params["char_file"])
  if network_type == "cnn":
    model = CNNNetwork(max_sentence_length=train_data.max_sentence_len,
                       **params)
  elif network_type == "lstm":
    model = LSTMNetwork(**params)
  model.build()
  model.train(train, dev)

if __name__ == "__main__":
  main(sys.argv[1])