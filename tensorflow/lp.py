import numpy as np
import os
from params import params
from data_processor import ATEDataProcessor
from stanfordcorenlp import StanfordCoreNLP

print("Connecting to CoreNLP server.....")
nlp = StanfordCoreNLP("{}/stanford-corenlp".format(os.path.expanduser("~")))
print("Connected!")
train_data = ATEDataProcessor(params["train_file"],
                              pos_source_file=params.get("pos_train_file"),
                              small_pos=params.get("small_pos"))
test_data = ATEDataProcessor(params["test_file"],
                             pos_source_file=params.get("pos_test_file"),
                             small_pos=params.get("small_pos"))

sentences = test_data.raw_sentences
dependencies = []
pos = []
i = 0
j = len(sentences)
'''
opinion_words = []
with open(params["opinion_words_file"]) as f:
     for line in f:
         word = line.split("\n")
         opinion_words.append(word)

stop_words = []
with open(params["stop_words_file"]) as f:
  for line in f:
    word = line.split("\n")
    stop_words.append(word)
'''

def dep_sentences(dep_list, pos_list, annotated_list):
  '''
  getting the dependency relations of the form
  (word_1, tag) 'relation' (word_2, tag)
  '''
  for i in range(len(dep_list)):
    a,b,c = dep_list[i]
    #print a,b,c
    word_1 = pos_list[b-1]
    #print word_1
    word_2 = pos_list[c-1]
    #print word_2
    ann_line = []
    ann_line.append(word_1)
    ann_line.append(a)
    ann_line.append(word_2)
    annotated_list.append(ann_line)    
  print "annotated list printing"
  print annotated_list

def get_aspects(annotated_list):
  aspects_set = set()
  aspect_words=[]
  for dep in annotated_list:
    (a,b,c)=dep
    (u,v)=a  #("word", "type") on left
    (x,y)=c  #("word", "type") on right
    if b =="amod" or b=="pnmod" or b=="subj" or b=="dobj" or b=="nsubj":
      if(v=="NN" or v=="NNS" and u not in stop_words):
          if x in opinion_words:
            aspects_set.add(u)
          elif( y=="VB"):
            aspects_set.add(u)
          elif(y=="JJS" or y=="JJ"):
            aspects_set.add(u)
      if(y=="NN" or y=="NNS" or y=="NNP" and x not in stop_words):
          if u in opinion_words:
            aspects_set.add(x)
          elif(v=="JJ" or v=="JJS"):
            aspects_set.add(x)
          elif(v=="VB" or v=="VBG" or v=="VBP"):
            if x not in stop_words:
              aspects_set.add(x)
    if(b=="conj"):
      if(u in aspects_set):
          aspects_set.add(x)
      elif(x in aspects_set):
          aspects_set.add(u)
    if(b == "compound" and (v == "NN" or v == "NNS" and y == "NN" or y =="NNS")):
            #print x + " " + u
      aspects_set.add(x+" "+u)
  for words in aspects_set:
    aspect_words.append(words.split(" "))
  print aspect_words

for sentence in sentences:
  print i, j
  i += 1
  dep_list = nlp.dependency_parse(sentence)
  pos_list = nlp.pos_tag(sentence)
  #print pos_list
  #print dep_list
  annotated_list = []
  print sentence
  dep_sentences(dep_list,pos_list,annotated_list)
  #get_aspects(annotated_list)

nlp.close()
