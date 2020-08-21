"""
Module to save dependencies of each sentence in given dataset to a npy array
"""
import numpy as np
import os
from cnn_params import params, data_set
from data_processor import ATEDataProcessor
from stanfordcorenlp import StanfordCoreNLP

print("Connecting to CoreNLP server..")
nlp = StanfordCoreNLP("{}/stanford-corenlp".format(os.path.expanduser("~")))
print("Connected!")
train_data = ATEDataProcessor("./train_data/{}_train_rules.xml".format(data_set), **params)
test_data = ATEDataProcessor("./test_data/{}_test_rules.xml".format(data_set), **params)

sentences = train_data.raw_sentences + test_data.raw_sentences
dependencies = []
pos = []
# ner = []
i = 0
j = len(sentences)
for sentence in sentences:
  print i, j
  i += 1
  dependencies.append(list(nlp.dependency_parse(sentence)))
  pos.append(list(nlp.pos_tag(sentence)))
#  ner.append(list(nlp.ner(sentence)))

nlp.close()
np.save(params["dependencies"], dependencies)
np.save(params["dependencies_pos"], pos)
# np.save(params["dependencies_ner"], ner)

