"""
This module does aspect term extraction using language pattern rules.
"""
import numpy as np
from cnn_params import params
from data_processor import ATEDataProcessor
from sklearn.metrics import precision_recall_fscore_support as score

def is_noun(tag):
  """
  A helper function to check if a given tag corresponds to noun
  args:
    tag (str): The pos tag
  returns:
    True if the tag corresponds to a noun
  """
  if tag == "NN" or tag == "NNS" or tag == "NNP" or tag == "NNPS":
    return True
  return False


def is_verb(tag):
  """
  Same as above but with verb instead
  """
  if (tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or 
      tag == "VBP" or tag == "VBZ"):
    return True
  return False


def get_dependendents(word_id, dependencies, dependency):
  """
  Given a word_id, and a dependency, find all the words which are dependenant on
  the given word with the given dependency from all dependencies given in 
  dependencies
  args:
    word_id (int): The word_id of the word in the sentence. Starts at 1.
    dependencies (list of tuples): Output of stanford dependency parser for the
      sentence
    dependency (list): The list dependencies which we have to search for
  return:
    dependent_words (list): The word ids of the word dependent on given word
  """
  dependent_words = []
  for _dependency, word, dependent in dependencies:
    if word == word_id and _dependency in dependency:
      dependent_words.append(dependent)
  return dependent_words


def get_word_lists(sentiment_lexicon="./dependencies/sentiment_lexicon.txt",
                   stop_words="./dependencies/stop_words.txt"):
  """
  Stores all opinion words (sentiment lexicon) and stop words in the given files
  to lists.
  Args:
    sentiment_lexicon (str): The path to the file containing sentiment lexicon
    stop_words (str): The path to the file containing stop_words
  Returns:
    lexicon (list): The list of sentiment words present in the given file.
    s_words (list): The list of stop words present in the given file
  """
  lexicon = []
  s_words = []

  with open(sentiment_lexicon) as fp:
    for line in fp:
      lexicon.append(line.strip())
  with open(stop_words) as fp:
    for line in fp:
      s_words.append(line.strip())

  return lexicon, s_words


def get_aspect_terms():
  """
  We use three rules to get aspect terms:
    Rule 1: for a word t, find it's subject n. If has an adverbial or adjective
            modifier present in sentiment lexicon, then the subject is a aspect.
            This is used when the sentence has no auxillary verb
    Rule 2.1: if the word is noun, find a verb t for which the word is 
              subject. If t is modified by an adjective or adverb or adverbial
              clause modifier, the word is an aspect term. This is used when the 
              sentence has auxillary verb
    Rule 2.2: If the word is a verb and has a direct object not present 
              in lexicon, mark that object as aspect term if it's a noun
    Rule 3: If a noun h is a complement of a coplar verb, mark it as an 
            aspect
  After applying these rules we eliminate any stop word that has been tagged
  as aspect term.
  """
  dependencies = np.load(params["dependencies"])
  pos = np.load(params["dependencies_pos"])
  lexicon, stop_words = get_word_lists()
  auxillary_verbs = ["be", "can", "could", "do", "have", "may", "might", "must",
                     "ought", "shall", "should", "will", "would", "dare", 
                     "need"]
  aspect_tags = []
  for j, line in enumerate(pos):
    deps = dependencies[j]
    aspect_tags_temp = ["O"] * len(line)
    aux_verb_present = False
    for word, tag in line:
      if word.lower() in auxillary_verbs:
        aux_verb_present = True
        continue
    for i, (word, tag) in enumerate(line):
      if not aux_verb_present:
        # Rule 1
        dependents = get_dependendents(word_id=i+1, 
                                       dependency=["nsubj", "nsubjpass"],
                                       dependencies=deps)
        # nsubj: nominal subject
        # nsubjpass: passive nominal subject
        # advmod: adverb modifier
        # amod: adjectival modifier
        dependents = [dep for dep in dependents if is_noun(line[dep-1][1])]
        modifier_dependents = get_dependendents(word_id=i+1,
                                                dependency=["advmod", "amod"],
                                                dependencies=deps)
        for mod_dep in modifier_dependents:
          if line[mod_dep-1][0] in lexicon:
            for dep in dependents:
              aspect_tags_temp[mod_dep-1] = "B"

      if aux_verb_present:
        if is_verb(tag):
          # Rule 2.1
          dependents = get_dependendents(word_id=i+1, 
                                         dependency=["nsubj", "nsubjpass"],
                                         dependencies=deps)
          dependents = [dep for dep in dependents if is_noun(line[dep-1][1])]
          modifier_dependents = get_dependendents(word_id=i+1,
                                                  dependency=["advmod", "amod",
                                                              "advcl"],
                                                  dependencies=deps)
          if modifier_dependents:
            for dep in dependents:
              aspect_tags_temp[dep-1] = "B"

          # Rule 2.2: 
          dependents = get_dependendents(word_id=i+1, dependency=["dobj"],
                                         dependencies=deps)
          # dobj: direct object
          dependents = [dep for dep in dependents if is_noun(line[dep-1][1])]
          for dep in dependents:
            if line[dep-1][0] not in lexicon:
              aspect_tags_temp[dep-1] = "B"
      # Rule 3
      cop_dependents = get_dependendents(word_id=i+1, dependency=["cop"], 
                                         dependencies=deps)
      # cop: copular verb
      if cop_dependents:
        dependents = get_dependendents(word_id=i+1, 
                                       dependency=["nsubj", "nsubjpass"],
                                       dependencies=deps)
        for dep in dependents:
          if is_noun(line[dep-1][1]):
            aspect_tags_temp[dep-1] = "B"

    for i, tag in enumerate(aspect_tags_temp):
      if tag == "B" and line[i][0].lower() in stop_words:
        aspect_tags_temp[i] = "O"
    aspect_tags.append(aspect_tags_temp)
  return aspect_tags

if __name__ == "__main__":
  train = ATEDataProcessor(params["train_file"], **params)
  test = ATEDataProcessor(params["test_file"], **params)
  annotated_sentences = train.annotated_sentences + test.annotated_sentences
  actual_tags = []
  for sentence in annotated_sentences:
    for _, _tag, _ in sentence:
      actual_tags.append(_tag)
  tags = get_aspect_terms()
  tags = [tag for _tags in tags for tag in _tags]
  x = score(actual_tags, tags)
  print x
  import pdb; pdb.set_trace()
