"""
Defines DataIterator object which we yields words and tags to be used for
training and testing
"""
from vocab_builder import get_ids, load_vocab_file


def minbatch_generator(data, batch_size):
  """
  Generates batch sized data partitions from data
  Args:
    data (DataIterator): object of DataIterator
    batch_size (int): size of each batch
  Returns:
    x_batch (list): batch sized list of words
    y_batch (list): batch sized list of tags
  """
  x_batch = []
  y_batch = []
  pos_batch = []

  for x, y, pos in data:
    if len(x_batch) == batch_size:
      yield x_batch, y_batch, pos_batch
      x_batch = []
      y_batch = []
      pos_batch = []

    x = zip(*x)

    x_batch += [x]
    y_batch += [y]
    pos_batch += [pos]

  if len(x_batch) != 0:
    yield x_batch, y_batch, pos_batch


class DataIterator(object):
  """
  An object of this class can be used to yield tuples of form (words, tags)
  """
  def __init__(self, annotated_sentences, **kwargs):
    """
    initializer for DataIterator class
    Args:
      annotated_sentences (list): an ATEDataProcessor.annotated_sentences like
        list.
      kwags:
        word_file (str): path to the file containing our word vocabulary
        char_file (str): path to the file containing our character vocabulary
    """
    self.annotated_sentences = annotated_sentences
    self.word_vocab = load_vocab_file(kwargs["word_file"])
    self.char_vocab = load_vocab_file(kwargs["char_file"])
    self.tags = {"O": 0, "B": 1, "I": 2}
    self.length = None

  def __iter__(self):
    for sentence in self.annotated_sentences:
      words = []
      tags = []
      pos = []
      for word, tag, pos_id in sentence:
        words.append(get_ids(word, self.word_vocab, self.char_vocab))
        tags.append(self.tags[tag])
        pos.append(pos_id)
      yield words, tags, pos

  def __len__(self):
    """
    Iterates over our dataset once and stores the length
    Returns:
      int: length of our dataset
    """
    if self.length is None:
      self.length = 0
      for _ in self:
        self.length += 1
    return self.length
