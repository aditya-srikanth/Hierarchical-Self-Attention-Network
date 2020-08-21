"""
A module to define our mappings from POS to ID
"""

class POSParser(object):
  # POS_MAP.keys() are the POS tags used by stanford POS tagger.
  POS_MAP = {'CC': 0, 'CD': 1, 'DT': 2, 'EX': 3, 'FW': 4, 'IN': 5, 'JJ': 6,
             'JJR': 7, 'JJS': 8, 'LS': 9, 'MD': 10, 'NN': 11, 'NNP': 13,
             'NNPS': 14, 'NNS': 12, 'PDT': 15, 'POS': 16, 'PRP': 17, 'PRP$': 18,
             'RB': 19, 'RBR': 20, 'RBS': 21, 'RP': 22, 'SYM': 23, 'TO': 24,
             'UH': 25, 'VB': 26, 'VBD': 27, 'VBG': 28, 'VBN': 29, 'VBP': 30,
             'VBZ': 31, 'WDT': 32, 'WP': 33, 'WP$': 34, 'WRB': 35}

  # This mapping will map the above keys to one of the noun(0), verb(1),
  # adjective(2), adverb(3), preposition(4), conjunction(5) misc(6)
  POS_MAP_SMALL = {'CC': 5, 'CD': 2, 'DT': 2, 'EX': 0, 'FW': 6, 'IN': 4, 'JJ': 2,
                   'JJR': 2, 'JJS': 2, 'LS': 6, 'MD': 0, 'NN': 0, 'NNP': 0,
                   'NNPS': 0, 'NNS': 0, 'PDT': 6, 'POS': 6, 'PRP': 0,
                   'PRP$': 0, 'RB': 3, 'RBR': 3, 'RBS': 3, 'RP': 6,
                   'SYM': 6, 'TO': 4, 'UH': 6, 'VB': 1, 'VBD': 1,
                   'VBG': 0, 'VBN': 1, 'VBP': 1, 'VBZ': 1, 'WDT': 2,
                   'WP': 0, 'WP$': 0, 'WRB': 3}
