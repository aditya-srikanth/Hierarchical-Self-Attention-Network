"""
Module to generate pos files
"""
import os
from stanfordcorenlp import StanfordCoreNLP
from data_processor import ATEDataProcessor

print("Connecting to CoreNLP server..")
nlp = StanfordCoreNLP("{}/stanford-corenlp".format(os.path.expanduser("~")))
print("Connected!")

if not os.path.isdir("./pos_files"):
  os.makedirs("./pos_files")


def store_pos(datafile, outfile):
  data_processor = ATEDataProcessor(datafile)
  with open(outfile, "w") as fp:
    nword = None
    for i, sentence in enumerate(data_processor.raw_sentences):
      print i, len(data_processor.raw_sentences)
      pos = list(nlp.pos_tag(sentence))
      for word, tag in pos:
        word = word.encode("utf-8")
        tag = tag.encode("utf-8")
        if word == tag or tag == "." or tag == "``" or tag == ":" or \
                tag == "''" or tag == "$" or tag == "-LRB-" or tag=="-RRB-":
          tag = "SYM"
        if word == "-LRB-":
          word = "("
          tag = "SYM"
        elif word == "-RRB-":
          word = ")"
          tag = "SYM"
        elif word == ":-RRB-":
          word = ":)"
          tag = "SYM"
        elif word == ":-LRB-":
          word = ":("
          tag = "SYM"
        elif " " in word:
          word, nword = word.split(" ")
        fp.write("{} {}\n".format(word, tag))
        if nword:
          fp.write("{} {}\n".format(nword, tag))
          nword = None
      fp.write("\n") 

store_pos("./train_data/lap_train.xml", "./pos_files/lap_train_pos.txt")
store_pos("./test_data/lap_test.xml", "./pos_files/lap_test_pos.txt")
store_pos("./train_data/rest_train.xml", "./pos_files/rest_train_pos.txt")
store_pos("./test_data/rest_test.xml", "./pos_files/rest_test_pos.txt")
nlp.close()
