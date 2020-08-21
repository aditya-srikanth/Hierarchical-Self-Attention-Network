import shelve
from crf_params import params, data_set
from data_processor import ATEDataProcessor

train = ATEDataProcessor(data_file=params["train_file"], **params)
test = ATEDataProcessor(data_file=params["test_file"], **params)
sentences = train.annotated_sentences + test.annotated_sentences
db = shelve.open("word_counts/word_counts_{}.db".format(data_set))
for i, sentence in enumerate(sentences):
  print i, len(sentences)
  for word, _, _ in sentence:
    word = word.encode("utf-8")
    word = str(word)
    db[word] = db.get(word, 0) + 1
db.close()
