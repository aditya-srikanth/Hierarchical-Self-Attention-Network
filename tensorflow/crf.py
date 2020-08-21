"""
Implements crf with various features.
"""
from __future__ import division
import pycrfsuite
import numpy as np
import shelve
import spacy
from pos_parser import POSParser
from data_processor import ATEDataProcessor
from crf_params import params, data_set
from vocab_builder import get_aspect_chunks
from sklearn.metrics import classification_report

nlp = spacy.load("en")

word_counts = shelve.open("word_counts/word_counts_{}.db".format(data_set))
dep_vecs = np.load(params["dependencies"])
pos_vecs = np.load(params["dependencies_pos"])
ner_vecs = np.load(params["dependencies_ner"])
pos_map = {index: tag for tag, index in POSParser.POS_MAP.items()}

def get_pos(pos_vec):
  """
  Gets raw pos_tag from pos_vec
  Args:
    pos_vec (list): A one shot vector that has 1 as its index for the
      corresponding tag.

  Returns:
     (str) The pos tag
  """
  return pos_map[np.argmax(pos_vec)]


def get_features(sentence, sent_count):
  """
  this function returns a list of dictionaries. Each dictionary represents
  features for a given word. This function is meant to call all other
  functions which will give it the required features.
  Args:
    sentence (list of tuples): This is a structure of form
      ATEDataProcessor.annotated_sentences

  Returns:
    a list of feature dictionaries

  """
  features_list = []
  words = [word for (word, _, _) in sentence]
  sent = unicode(" ".join(words))
  doc = nlp(sent)
  noun_phrases = []
  head_words = []
  for w in doc:
    head_words.append((w.head, w.head.pos_))
  for phrase in doc.noun_chunks:
    noun_phrases.extend(phrase.text.split(" "))
  for i, (word, _, _) in enumerate(sentence):
    word = word.encode("utf-8")
    ner = ner_vecs[sent_count]
    pos = pos_vecs[sent_count]
    deps = dep_vecs[sent_count]
    dep1_list = []
    dep2_list = []
    for relation, sub, ob in deps:
      if relation != "root":
        if sub == word and (relation == "amod" or relation == "nsubj"
                             or relation == "dep"):
          dep1_list.append(relation)
        if ob == word and (relation == "nsubj" or relation == "dobj"
                             or relation == "dep"):
          dep2_list.append(relation)

    if i >= len(ner):
      print "in here"
      ner_temp = [(word, tag) if " " not in word 
                  else [(word.split(" ")[0], tag), (word.split(" ")[1], tag)] 
                  for word, tag in ner]
      pos_temp = [(word, tag) if " " not in word 
                  else [(word.split(" ")[0], tag), (word.split(" ")[1], tag)] 
                  for word, tag in pos]
      ner = []
      pos = []

      for w in ner_temp:
        if type(w) == list:
          ner.append(w[0])
          ner.append(w[1])
        else:
          ner.append(w)
      for w in pos_temp:
        if type(w) == list:
          pos.append(w[0])
          pos.append(w[1])
        else:
          pos.append(w)

    pos = pos[i][1]  
    ner = ner[i][1]
    features = [
      "bias",
      "word.lower=" + word.lower(),
      "word.isupper={}".format(word.isupper()),
      "word.istitle={}".format(word.istitle()),
      "word.isdigit={}".format(word.isdigit()),
      "postag=" + pos,
      "ner={}".format(ner),
      "wordcount={}".format(word_counts[str(word)]),
      "headword={}".format(head_words[i][0]),
      "headword_pos={}".format(head_words[i][1]),
      "part_of_nounchunk={}".format(word in noun_phrases),
      "dep1_list={}".format(dep1_list),
      "dep2_list={}".format(dep2_list)
    ]
    if i > 0:
      word_prev, _, pos_id_prev = sentence[i-1]
      word_prev = word_prev.encode("utf-8")
      pos_prev = pos_vecs[sent_count][i-1][1] 
      ner_prev = ner_vecs[sent_count][i-1][1]
      features.extend([
        "-1:word.lower={}".format(word_prev.lower()),
        "-1:word.istitle={}".format(word_prev.istitle()),
        "-1:word.isupper={}".format(word_prev.isupper()),
        "-1:postag={}".format(pos_prev),
        "-1:ner={}".format(ner_prev),
        "-1:headword={}".format(head_words[i-1][0]),
        "-1:headword_pos={}".format(head_words[i-1][1])
      ])
    else:
      features.append("BOS")

    if i < len(sentence) - 1:
      word_next, _, pos_id_next = sentence[i+1]
      word_next = word_next.encode("utf-8")
      ner = ner_vecs[sent_count]
      pos = pos_vecs[sent_count]
      
      if i+1 >= len(ner):
        print "in here"
        ner_temp = [(word, tag) if " " not in word 
                    else [(word.split(" ")[0], tag), (word.split(" ")[1], tag)] 
                    for word, tag in ner]
        pos_temp = [(word, tag) if " " not in word 
                    else [(word.split(" ")[0], tag), (word.split(" ")[1], tag)] 
                    for word, tag in pos]
        ner = []
        pos = []

        for w in ner_temp:
          if type(w) == list:
            ner.append(w[0])
            ner.append(w[1])
          else:
            ner.append(w)
        for w in pos_temp:
          if type(w) == list:
            pos.append(w[0])
            pos.append(w[1])
          else:
            pos.append(w)

      ner_next = ner[i+1][1] 
      pos_next = pos[i+1][1]
      features.extend([
        "+1:word.lower={}".format(word_next.lower()),
        "+1:word.istitle={}".format(word_next.istitle()),
        "+1:postag={}".format(pos_next),
        "+1:ner={}".format(ner_next),
        "+1:headword={}".format(head_words[i+1][0]),
        "+1:headword_pos={}".format(head_words[i+1][1])
      ])
    else:
      features.append("EOS")
    features_list.append(features)
  return features_list


def get_labels(sentence):
  return [label for _, label, _ in sentence]

train = ATEDataProcessor(data_file=params["train_file"], **params)
test = ATEDataProcessor(data_file=params["test_file"], **params)

x_train = [get_features(s, i) for i, s in enumerate(train.annotated_sentences)]
y_train = [get_labels(s) for s in train.annotated_sentences]
train_count = len(train.annotated_sentences) 
x_test = [get_features(s, i+train_count)
          for i, s in enumerate(test.annotated_sentences)]
y_test = [get_labels(s) for s in test.annotated_sentences]
word_counts.close()

trainer = pycrfsuite.Trainer(verbose=True)
for x, y in zip(x_train, y_train):
  trainer.append(x, y)
# todo: do hyperparameter tuning on this
trainer.set_params({
  "c1": 0.01,
  "c2": 0.1,
  "max_iterations": 100,
  "feature.possible_transitions": True
})
trainer.train("crf.model")
tagger = pycrfsuite.Tagger()
tagger.open("crf.model")
y_pred = [tagger.tag(xseq) for xseq in x_test]
tags = {"O": 0, "B": 1, "I": 2}
tp = 0
tn = 0
fp = 0
fn = 0
for i, label_actual in enumerate(y_test):
  label_actual = [tags[tag] for tag in label_actual]
  label_pred = [tags[tag] for tag in y_pred[i]]
  a_terms_actual, non_a_terms_actual = get_aspect_chunks(label_actual)
  for a_term in a_terms_actual:
    a_term_pred = []
    for index in a_term:
      if not label_actual[index] == label_pred[index]:
        fn += 1
        break
    else:
      tp += 1
  for index in non_a_terms_actual:
    if label_actual[index] == label_pred[index]:
      tn += 1
    else:
      fp += 1 
precision = 0 if tp+fp == 0 else tp/(tp+fp)
recall = 0 if tp+fn == 0 else tp/(tp+fn)
fscore = 0 if precision+recall == 0 else (2*recall*precision)/(recall+precision) 

print precision, recall, fscore   

