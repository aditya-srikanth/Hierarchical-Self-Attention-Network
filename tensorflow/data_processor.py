"""
A module to pre process data.
"""
import os
import sys
import xml.etree.ElementTree as ET

from nltk.tokenize import word_tokenize
from pos_parser import POSParser


class ATEDataProcessor(object):
  """
  Given an XML file of the structure <sentences> <sentence> <text> </text>
  <aspectTerms> <aspectTerm/> </aspectTerms> </sentence> </sentences>,
  this class deals with parsing the data
  """
  def __init__(self, data_file, pos_id=0, **kwargs):
    """
    __init__ for ATEDataProcessor
    Args:
      data_file (str): The file path of the xml file which has our training data
      pos_id (int, 0): The starting pos_id of the first word in the dataset.
      kwargs:
        use_pos (Boolean): Whether or not to use pos_files
        pos_source_file (str): The pos file to be used if use_pos is true
        small_pos (Boolean): Whether to use pos of 6 dimensions or 36 dimensions
        use_text_dataset (Boolean): If true, the ate and pos tags are obtained 
          from datafile, considering it to be a text file of format 
          (word, ATE-Tag, POS-Tag)
        pos_train_file (str): If "train" is in the name of data_file, this file 
          will be used to get pos if use_pos=True
        pos_test_file (str): If "test" is in the name of data_file, this file 
          will be used to get pos if use_pos = True

    """
    self.use_pos = kwargs.get("use_pos", False)
    self.pos_id = pos_id
    self.data_file = data_file
    self.raw_sentences = []
    self.annotated_sentences = []
    self.pos_words = []
    self.pos_tags = []
    self.word_vocab = set()
    self.char_vocab = set()
    self.max_sentence_len = 0
    self.max_word_len = 0
    self.small_pos = kwargs.get("small_pos", False)
    self.use_ate_source_file = kwargs.get("use_text_dataset", False)
    if "train" in self.data_file:
      self.pos_source_file = kwargs.get("pos_train_file")
    else:
      self.pos_source_file = kwargs.get("pos_test_file")
    if self.use_ate_source_file:
      self.ate_source_file = data_file
    else:
      self.tree = ET.parse(self.data_file)
      self.ate_source_file = None
    if self.use_pos and (self.pos_source_file is not None or
                         self.ate_source_file is not None):
      self.pos_lines = []
      self.get_pos_lines()
      self.use_pos = True
    else:
      self.use_pos = False

    if kwargs.get("tokenizer") == "stanford":
      from stanfordcorenlp import StanfordCoreNLP
      print("Connecting to Server")
      self.nlp = StanfordCoreNLP("{}/stanford-corenlp".format(
        os.path.expanduser("~")
      ))
      self.tokenizer = self.nlp.word_tokenize
      print("Connected to server")
    else:
      self.tokenizer = word_tokenize
      self.nlp = None

    if self.ate_source_file:
      self.process_data_file()
    else:
      self.process_data()

  def get_pos_lines(self):
    """
    Read pos file and get pos words and corresponding tags.
    """
    if self.small_pos:
      # if we want to use only 6 pos tags
      print("Using condensed pos tags!")
      mapping = POSParser.POS_MAP_SMALL
    else:
      mapping = POSParser.POS_MAP

    if self.ate_source_file:
      # if we are given a text dataset of the format "word ATE-Tag POS-Tag"
      print("Using text dataset to build dataset!")
      index = 2
      pos_file = self.ate_source_file
    elif self.pos_source_file:
      print("Using pos files to build dataset!")
      pos_file = self.pos_source_file
      index = 1
    else:
      raise Exception("either pass a pos_source_file or ate_source_file "
                      "argument")

    with open(pos_file) as fp:
      words = []
      for line in fp:
        if line == "\n":
          self.pos_words.append(words)
          words = []
        else:
          words.append(line.strip().split(" ")[0])
          tag = line.strip().split(" ")[index]
          pos_vec = [0] * len(mapping)
          pos_vec[mapping[tag]] = 1
          self.pos_tags.append(pos_vec)
      if words:
        # we shouldn't reach here. If we reached here, that means there is
        # something wrong with out data set.
        raise Exception("There seems to be something wrong with the pos file "
                        "you're using. May be it is missing a \n character at "
                        "the end?")

  def process_data_file(self):
    """
    This method is to process data when data format is a text file of the format
    word tag pos_tag, instead of xml file.
    Returns:
      None
    """
    print("Getting ate and pos data from a text file.")
    words = []
    annotated_words = []
    with open(self.ate_source_file) as fp:
      for line in fp:
        if line == "\n":
          self.raw_sentences.append(" ".join(words))
          self.annotated_sentences.append(annotated_words)
          self.word_vocab.update(set(words))
          if len(words) > self.max_sentence_len:
            self.max_sentence_len = len(words)
          words = []
          annotated_words = []
        else:
          word, ate, pos = line.strip().split(" ")
          words.append(word)
          annotated_words.append((word, ate, self.pos_id))
          self.pos_id += 1
          self.char_vocab.update(set(word))

  def process_data(self):
    """
    The main function which processes data. This populates self.raw_sentences and
    self.annotated_sentences lists. self.raw_sentences has raw sentences,
    where as the other list will have a list of tuples.
    eg: a list of [(I, O), (like, O), (battery, B), (life, B)]

    O is given as tag for all non aspect terms
    B is given as tag for all words belonging to the aspect term. we aren't
    using the "I" tag.

    We use pos sentences and by extension stanford tokenizer if pos files are
    available

    """
    print("Getting data from xml file..")
    root = self.tree.getroot()
    sentences = root.findall('sentence')
    p = 0
    for sentence in sentences:
      text = sentence.find("text").text
      text = text.encode("utf-8").strip()
      self.raw_sentences.append(text)
      words = self.tokenizer(text.decode("utf-8"))
      if self.use_pos:
        words = self.pos_words[p]
        if sys.version_info[0] < 3:
          words = [word.decode("utf-8") for word in words]
        p += 1
      for w in words:
        if len(w) > self.max_word_len:
          self.max_word_len = len(w)
        self.char_vocab.update(set(w))
      self.word_vocab.update(set(words))
      tags = ["O"]*len(words)
      pos_ids = [0] * len(words)
      for i in range(len(words)):
        pos_ids[i] = self.pos_id
        self.pos_id += 1
      curr_len = len(words)
      if curr_len > self.max_sentence_len:
        self.max_sentence_len = curr_len
      aspect_terms = sentence.find("aspectTerms")
      if aspect_terms is not None:
        aspect_terms = aspect_terms.findall("aspectTerm")
        for term in aspect_terms:
          a_terms = term.attrib['term'].split(" ")
          indices = [i for i, e in enumerate(words) if e in a_terms]
          for i, e in enumerate(indices):
            if i == 0:
              tags[e] = "B"
            else:
              tags[e] = "I"
      self.annotated_sentences.append(list(zip(words, tags, pos_ids)))
    if self.nlp:
      self.nlp.close()

  def write_to_file(self, outfile, delimiter=" "):
    """
    Write the annotated data set in the standard format. each line has a word
    and its tag separatd by the specified delimiter and each sentence is
    separated from another by an empty line.
    Args:
      outfile (str): The name of the file to which data should be written
      delimiter (" ", str): The delimiter between the word and its tag.
    Returns:
      None
    """
    print("writing to {}".format(outfile))
    with open(outfile, "w") as fp:
      for sentence in self.annotated_sentences:
        for word, tag, _ in sentence:
          fp.write("{}{}{}\n".format(word, delimiter, tag))
        fp.write("\n")

