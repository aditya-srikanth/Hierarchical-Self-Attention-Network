"""
A helper script to generate raw text files from xml files.
"""
from data_processor import ATEDataProcessor
from params import params

if __name__ == "__main__":
  # escape_word is the character that seperates two sentences in the final
  # output. Remember that some of the raw sentences don't end with
  # punctuation, don't begin with capital letters, so it might be better to
  # have a end of sentence symbol.
  escape_word = "\n"
  dp_train = ATEDataProcessor(params["train_file"])
  dp_test = ATEDataProcessor(params["test_file"])
  raw_sentences = dp_train.raw_sentences + dp_test.raw_sentences
  raw_sentences = escape_word.join(raw_sentences)
  with open("raw_rest.txt", "w")as fp:
    fp.write(raw_sentences)
