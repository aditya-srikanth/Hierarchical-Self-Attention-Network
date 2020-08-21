"""
Implements crf with various features.
"""
import numpy as np
from pos_parser import POSParser
from data_processor import ATEDataProcessor
from crf_params import params
from sklearn_crfsuite import CRF, metrics, scorers

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
  for i, (word, _, pos_id) in enumerate(sentence):
    word = word.encode("utf-8")
    pos = pos_vecs[sent_count][i][1]  
    ner = ner_vecs[sent_count][i][1]
    features = {
      "bias": 1.0,
      "word.lower": word.lower(),
      "word[-3:]": word[-3:],
      "word[-2:]": word[-2:],
      "word.isupper": word.isupper(),
      "word.istitle": word.istitle(),
      "word.isdigit": word.isdigit(),
      "postag" : pos,
      "postag[:2]" : pos[:2],
      "ner": ner
    }
    if i > 0:
      word_prev, _, pos_id_prev = sentence[i-1]
      word_prev = word_prev.encode("utf-8")
      pos_prev = pos_vecs[sent_count][i-1][1] 
      ner_prev = ner_vecs[sent_count][i-1][1]
      features.update({
        "-1:word.lower": word_prev.lower(),
        "-1:word.istitle": word_prev.istitle(),
        "-1:word.isupper": word_prev.isupper(),
        "-1:postag": pos_prev,
        "-1:postag[:2]": pos_prev[:2],
        "-1:ner": ner_prev
      })
    else:
      features["BOS"] = True

    if i < len(sentence) - 1:
      word_next, _, pos_id_next = sentence[i+1]
      word_next = word_next.encode("utf-8")
      ner_next = ner_vecs[sent_count][i+1][1] 
      pos_next = pos_vecs[sent_count][i+1][1] 
      features.update({
        "+1:word.lower": word_next.lower(),
        "+1:word.istitle": word_next.istitle(),
        "+1:word.isupper": word_next.isupper(),
        "+1:postag": pos_next,
        "+1:postag[:2]": pos_next[:2],
        "+1:ner": ner_next
      })
    else:
      features["EOS"] =  True
    features_list.append(features)
  return features_list


def get_labels(sentence):
  return [label for _, label, _ in sentence]

train = ATEDataProcessor(data_file=params["train_file"], **params)
test = ATEDataProcessor(data_file=params["test_file"], **params)

x_train = [get_features(s, i) for i, s in enumerate(train.annotated_sentences)]
y_train = [get_labels(s) for s in train.annotated_sentences]
train_count = len(train.annotated_sentences) 
x_test = [get_features(s, i+train_count) for i, s in enumerate(test.annotated_sentences)]
y_test = [get_labels(s) for s in test.annotated_sentences]

crf = CRF(algorithm="lbfgs", c1=1.0, c2=0.1, max_iterations=150,
          all_possible_transitions=True)
crf.fit(x_train, y_train)
labels = ["O", "B"]
y_pred = crf.predict(x_test)
print(metrics.flat_classification_report(y_test, y_pred, labels=labels,
                                         digits=3))
