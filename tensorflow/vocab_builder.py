"""
Module which defines methods that facilitates building the vocabulary (words and
characters) along with word vectors.
"""
import sys
import numpy as np
from copy import deepcopy


def get_word_vector_voacb(vector_file):
  """
  Return all the words for which we have vectors available.
  Args:
    vector_file (str): The path to the file where word vectors are present

  Returns:
    set : set of words for which we have word vectors
  """
  words = set()
  with open(vector_file) as fp:
    for line in fp:
      word = line.strip().split(" ")[0]
      words.add(word)
  return words


def dump_vocab(data_objs, word_file, char_file, vector_file):
  """
  Write word vocabulary and character vocabulary from given list of
  ATEDataProcessor objects.
  Args:
    data_objs (list): List of ATEDataProcessor objects
    word_file (str): The name of the file to store our words
    char_file (str): The name of the file to store our characters
    vector_file (str): The name of the file with avilable word vectors

  Returns:
    tuple of dictionaries:
      word_to_id (dict): A mapping with key = word, value = id.
      char_to_id (dict): A mapping with key = char, value = id.
  """
  word_vocab = set()
  char_vocab = set()
  for obj in data_objs:
    word_vocab = word_vocab | obj.word_vocab
    char_vocab = char_vocab | obj.char_vocab

  print("there are {} words in our vocabulary".format(len(word_vocab)))
  vector_vocab = get_word_vector_voacb(vector_file)
  print("we have {} vectors".format(len(vector_vocab)))
  word_vocab = word_vocab & vector_vocab
  print("we have vectors for {} words".format(len(word_vocab)))
  word_vocab.add("<unknown>")
  word_vocab.add("<number>")

  word_to_id = dict()

  # write words.
  with open(word_file, "w") as fp:
    for _id, word in enumerate(word_vocab):
      fp.write("{}\n".format(word))
      word_to_id[word] = _id

  char_to_id = dict()

  # write characters.
  with open(char_file, "w") as fp:
    for _id, char in enumerate(char_vocab):
      if sys.version_info[0] < 3:
        char = char.encode("utf-8")
      fp.write("{}\n".format(char))
      char_to_id[char] = _id

  print("we have {} characters".format(len(char_to_id.keys())))
  return word_to_id, char_to_id


def build_raw_corpus(data_objs, out_file):
  """
  Build raw text corpus without any annotations. Can be used to build
  wordvectors using fasttext.
  Args:
    data_objs (list of ATEDataProcessor objects): each ATEDataProcessor object
      already has raw sentences stored in the object.raw_sentences variable.
      It's a simple matter of using them to build the corpus.
    out_file (str): The name of the file where this corpus should be written.
  Returns:
    None
  """
  corpus = ""
  for obj in data_objs:
    for sentence in obj.raw_sentences:
      corpus += sentence.strip()
      corpus += "\n"
  with open(out_file, "w") as fp:
    fp.write(corpus)


def store_word_vectors(vector_file, word_to_id, stored_vectors):
  """
  Stores the word vectors of our required words (the words in our vocabulary)
  into a numpy array. Index of word is obtained from word_to_id dictionary and
  at that index in the array the vector to the word is stored.
  Args:
    vector_file (str): The path to the file where word vectors are stored
    word_to_id (dict): a mapping with word => id mapping
    stored_vectors (str) : the path to where word vectors of our vocabulary are
      stored.
  Returns:
    None
  """
  print("Storing word vectors..")
  ii = 0
  vecs = None
  has = []
  with open(vector_file) as fp:
    for line in fp:
      word = line.strip().split(" ")[0]
      vectors = [float(vec) for vec in line.strip().split(" ")[1:] if vec]
      if ii == 0:
        dim = len(vectors)
        vecs = np.zeros([len(word_to_id), dim])
        ii += 1
      if word in word_to_id:
        vecs[word_to_id[word]] = vectors
        has.append(word)
#  for word in word_to_id:
#    if word not in has:
#        print word
  np.save(stored_vectors, vecs)
  print("Stored vectors in the file: {}".format(stored_vectors))


def get_ids(word, word_to_id, char_to_id):
  """
  Return the ids of the word and the ids of characters in the word. we need
  ids to look up for the word embeddings and character embeddings while training.
  Args:
    word (str): The word whose id is to be determined
    word_to_id (dict): A mapping of word => id
    char_to_id (dict): A mapping of char => id
  Returns:
    tuple (char_ids, word_id):
      char_ids (list) : a list of ids of individual characters in the word
      word_id (int) : id of the word
  """
  char_ids = []
  for char in word:
    if char in char_to_id:
      char_ids.append(char_to_id[char])
  word = word.lower()
  if word.isdigit():
    word_id = word_to_id["<number>"]
  elif word in word_to_id:
    word_id = word_to_id[word]
  else:
    word_id = word_to_id["<unknown>"]
  return char_ids, word_id


def load_vocab_file(filename):
  """
  Loads the vocab file into a dictionary. The line number is the id of the
  word or character in the line.
  Args:
    filename (str): The path to the vocab file
  Returns:
    id_dict (dict): A mapping from word => id
  """
  i = 0
  id_dict = {}
  with open(filename) as fp:
    for line in fp:
      id_dict[line.strip()] = i
      i += 1
  return id_dict

def get_aspect_chunks(labels):
  """
  Merge consecutive I's with the preceeding B to get an aspect term
  Args:
    labels (list): List of labels of words in a given sentence
  Returns:
    a_terms (list): list of aspect terms. each aspect term is list of indices of
      words corresponding to aspect term
    non_a_terms (list): list of indices corresponding to non aspect terms
  """
  a_terms = []
  non_a_terms = []
  curr_term = []
  chunk_start = False
  for i, e in enumerate(labels):
    if not chunk_start:
      if e == 1 or e == 2:
        chunk_start = True
        curr_term.append(i)
        continue
    else:
      if e == 0:
        a_terms.append(deepcopy(curr_term))
        curr_term = []
        chunk_start = False
      elif e == 2:
        curr_term.append(i)
      elif e == 1:
        a_terms.append(deepcopy(curr_term))
        curr_term = []
        curr_term.append(i)
  if curr_term:
    a_terms.append(deepcopy(curr_term))
  for i, _ in enumerate(labels):
    if not any(i in terms for terms in a_terms):
      non_a_terms.append(i)
  return a_terms, non_a_terms
