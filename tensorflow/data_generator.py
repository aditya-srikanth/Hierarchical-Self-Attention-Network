import sys
import numpy as np
from data_processor import ATEDataProcessor
from vocab_builder import store_word_vectors, dump_vocab


def generate_data(kwargs):
  """
  Generate data based on params given as input
  Args:
      kwargs (dict): Params required for building data set.
  """
  train = ATEDataProcessor(
    data_file=kwargs["train_file"], **kwargs
  )
  test = ATEDataProcessor(
    data_file=kwargs["test_file"], **kwargs
  )
  w_dim = kwargs["w_dim"]
  vector_text_file = kwargs["raw_vectors"]
  word_to_id, _ = dump_vocab([train, test], kwargs["word_file"],
                             kwargs["char_file"], vector_text_file)
  store_word_vectors(vector_file=vector_text_file, word_to_id=word_to_id,
                     stored_vectors=kwargs["vector_file"])
  if kwargs.get("multi_rnn") or kwargs.get("use_additional_embeddings"):
    vector_text_file_2 = kwargs["raw_vectors_2"]
    store_word_vectors(vector_file=vector_text_file_2, word_to_id=word_to_id, 
                       stored_vectors=kwargs["vector_file_2"])
  if kwargs.get("use_pos"):
    print("building pos embeddings array")
    train_tags = np.array(train.pos_tags)
    test_tags = np.array(test.pos_tags)
    tags = np.concatenate([train_tags, test_tags])
    np.save(kwargs["pos_file"], tags)

if __name__ == "__main__":
  if sys.argv[1] == "lstm":
    from params import params
  elif sys.argv[1] == "cnn":
    from cnn_params import params
  generate_data(params)
