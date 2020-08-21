import codecs
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


def get_word_vectors(a_words):
  a_words = set(a_words)
  vecs = []
  words = []
  with codecs.open("glove.6B.100d.txt", encoding="utf-8") as fp:
    for line in fp:
      comps = line.strip().split(" ")
      if comps[0] in a_words:
        if comps[0] not in words:
          words.append(comps[0])
          vecs.append(np.array([float(i) for i in comps[1:]]))
  return words, vecs

def find_word_clusters(words, labels):
  cluster_to_words = [[] for _ in set(labels)]
  for i, e in enumerate(labels):
    cluster_to_words[e].append(words[i])
  return cluster_to_words

if __name__ == '__main__':
  for n_clusters in xrange(5, 6):
    a_terms = np.load("pred_aspect_words_rest.npy")
    words_, vecs = get_word_vectors(a_terms)
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
    kmeans.fit(vecs)
    cluster_labels = kmeans.labels_
    cluster_inertia = kmeans.inertia_
    cluster_to_words = find_word_clusters(words_, cluster_labels)
    print n_clusters, metrics.silhouette_score(vecs, cluster_labels, metric="euclidean")
    import pdb; pdb.set_trace()