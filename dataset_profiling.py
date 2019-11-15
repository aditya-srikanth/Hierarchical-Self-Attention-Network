try: 
    import xml.etree.cElementTree as ET 
except Exception as e:
    print('cElementTree not present')
    import xml.etree.ElementTree
import config
from data_utils import *

if __name__ == "__main__":
    vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path], store= config.mapping_file )
    train_dataset = ReviewDataset(config.dataset_path, preprocessed= False, vocab= vocab)
    test_dataset = ReviewDataset(config.test_dataset_path, preprocessed= False, vocab= vocab)
    train_list = train_dataset.get_review_list()
    test_list = test_dataset.get_review_list()

    no_aspect = 0
    with_aspect = 0
    total_aspect_lengths = []
    review_counts = {}
    for review in train_list:
        if review.aspect_terms is None:
            no_aspect += 1
        else:
            with_aspect += 1
            indicator = {}
            review_aspect_lengths = [ len( token ) for token in review.aspect_term_tokens]
            for length in review_aspect_lengths:
                if length not in indicator:
                    indicator[ length ] = True
                    if length in review_counts:
                        review_counts[ length ] += 1
                    else:
                        review_counts[ length ] = 1

            total_aspect_lengths += review_aspect_lengths
            
    print('\n\nTRAIN DATASET')
    print( 'with_aspect: ', with_aspect )
    print('without_aspect: ', no_aspect)
    print('%aspect: ', with_aspect/(with_aspect + no_aspect) * 100)
    print('average aspect length: ', sum(total_aspect_lengths)/len(total_aspect_lengths))
    frequency_size = { i: total_aspect_lengths.count(i) for i in range(1,max(total_aspect_lengths)+1) }
    print('aspect document frequencies: ', review_counts)
    print('aspect frequencies: ')
    print(frequency_size)
    

    no_aspect = 0
    with_aspect = 0
    total_aspect_lengths = []
    review_counts = {}
    for review in test_list:
        if review.aspect_terms is None:
            no_aspect += 1
        else:
            with_aspect += 1
            indicator = {}
            review_aspect_lengths = [ len( token ) for token in review.aspect_term_tokens]
            for length in review_aspect_lengths:
                if length not in indicator:
                    indicator[length] = True
                    if length in review_counts:
                        review_counts[ length ] += 1
                    else:
                        review_counts[ length ] = 1

            total_aspect_lengths += review_aspect_lengths
            
    print('\n\nTEST DATASET')
    print( 'with_aspect: ', with_aspect )
    print('without_aspect: ', no_aspect)
    print('%aspect: ', with_aspect/(with_aspect + no_aspect) * 100)
    print('average aspect length: ', sum(total_aspect_lengths)/len(total_aspect_lengths))
    frequency_size = { i: total_aspect_lengths.count(i) for i in range(1,max(total_aspect_lengths)+1) }
    print('aspect document frequencies: ', review_counts)
    print('aspect frequencies: ')
    print(frequency_size)
