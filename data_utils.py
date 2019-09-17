import config
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from torch import int32, float32
import numpy as np
import os
import torch

try: 
    import xml.etree.cElementTree as ET 
except Exception as e:
    print('cElementTree not present')
    import xml.etree.ElementTree


def evaluation_metrics(predictions, targets):
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    print( predictions, predictions.shape, targets.shape )
    print(sum((targets == predictions).float()))
    for i in range( len( predictions ) ):
        
        if targets[i] == 3: # special padding value, ignore this
            continue 

        elif predictions[i] == config.bio_dict['O'] and targets[i] == config.bio_dict['O']:
            tn += 1
        elif targets[i] == config.bio_dict['O'] and predictions[i] != config.bio_dict['O']:
            fp += 1
        elif targets[i] == config.bio_dict['B']: # B tag seen
            matched = True
            begin = i
            while   i < len( predictions ) and ( targets[i] != config.bio_dict['O'] or targets[i] != 3 ) and \
                    not ( i > begin and targets[i] == config.bio_dict['B'] ): # B tag not seen again
                
                if not matched:
                    i += 1
                elif targets[i] == predictions[i]:
                    i += 1
                elif targets[i] != predictions[i]:
                    matched = False
                    i += 1
            if matched:
                tp += 1
            else:
                fn += 1
            i -= 1
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    fscore = (2*recall*precision)/(recall+precision) if recall+precision else 0
    accuracy = (tp + tn)/ ( tp + tn + fp + fn)
    print('tp: ',tp, ' tn:', tn, ' fp: ', fp, ' fn: ', fn)
    metrics = dict()
    metrics['acc'] = accuracy
    metrics["p_1"] = precision
    metrics["f_1"] = fscore
    metrics["r_1"] = recall
    return metrics

def create_embedding_matrix( vocab, embedding_dim, device= config.device, dataset_path= None, save_weight_path= None):
    num_tokens = vocab.get_vocab_size()
    word2idx = vocab.get_vocab()
    embedding_matrix = np.zeros( ( num_tokens, embedding_dim ) )
    print('number of tokens: ', num_tokens, ' embedding dimensions: ', embedding_dim )
    print('embedding matrix shape: ',embedding_matrix.shape)
    if dataset_path != None:
        
        if os.path.isfile(save_weight_path):
            embedding_matrix = np.load( save_weight_path )
            print('retrieved embedding weights from existing file')
            return tensor( embedding_matrix, requires_grad= True, dtype= float32).to( device )
        
        if isinstance( dataset_path, str ):
        
            with open(dataset_path, 'r',encoding='utf-8') as f:
                num_mapped_words = 0
                num_total_words_seen = 0
                for line in f:
                    values = line.split()
                    word = values[0]
                    num_total_words_seen += 1
                    
                    if word in word2idx and not word == '<pad>' and not word == '<unk>':
                        vector = np.asarray( values[1:] )
                        embedding_matrix[ word2idx[ word ], : ] = vector
                        num_mapped_words += 1
                    
                    if num_total_words_seen % 1000000 == 0:
                        print('num_total_words_seen ',num_total_words_seen)

        elif isinstance( dataset_path, list ):
            for path in dataset_path:

                with open( path, 'r', encoding= 'utf-8' ) as f:

                    num_mapped_words = 0
                    num_total_words_seen = 0

                    for line in f:

                        values = line.split()
                        word = values[0]
                        num_total_words_seen += 1
                        
                        if word in word2idx and not word == '<pad>' and not word == '<unk>':
                            vector = np.asarray( values[1:] )
                            embedding_matrix[ word2idx[ word ], : ] = vector
                            num_mapped_words += 1
                        
                        if num_total_words_seen % 1000000 == 0:
                            print('num_total_words_seen ',num_total_words_seen)


        print('loaded pretrained matrix, ', ' num mapped words: ', num_mapped_words)
        if save_weight_path != None:
            np.save(save_weight_path, embedding_matrix)
        return tensor( embedding_matrix, requires_grad= True, dtype= float32).to(device)

    print('loaded trainable embedding matrix')
    
    return tensor( embedding_matrix, requires_grad= True, dtype= float32).to(device)

def parse_xml(file_path):

    review_list = []
    num_no_aspect_sentences = 0

    tree = ET.parse( file_path )
    root = tree.getroot()

    for child in root:
        sentence = child.find('text').text
        sentence = sentence.strip()
        sentence = re.sub(r'[\t]',' ',sentence)
        sentence = re.sub(r'[^a-zA-Z0-9 !@#$%^&*()-_=+~`\'\":;.,/?]', '',sentence).lower()
        
        
        for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
            sentence = sentence.replace(ch, " " + ch + " ")
            
        sentence = ' '.join( sentence.split() )

        aspect_terms = child.find('aspectTerms')

        if aspect_terms != None:
        
            aspects = []
            for i in aspect_terms:
                
                aspect = i.attrib['term'].strip()
                aspect = re.sub(r'\t',' ',aspect)
                aspect = re.sub(r'[^a-zA-Z0-9 !@#$%^&*()-_=+~`\'\":;.,/?]', '', aspect).lower()
                
                for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
                    aspect = aspect.replace(ch, " " + ch + " ")
                
                aspect = ' '.join( aspect.split() )

                aspects.append( aspect )
                
            review_list.append( Review( child.attrib['id'], sentence, aspects ) )
        else:
                num_no_aspect_sentences += 1
                review_list.append( Review( child.attrib['id'], sentence ) )
        
    print('Number of no-aspect sentences = ', num_no_aspect_sentences )
    print('Number of total reviews = ', len( root ) )
    
    return review_list

def subfinder(mylist, pattern):
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            return ( i, i+len(pattern) - 1 )

def generate_bio_tags( start_end_indices, max_length):

    bio_tags = tensor([ config.bio_dict['O'] ] * max_length, dtype= torch.int64) 
    
    if start_end_indices == None or len( start_end_indices ) == 0:
        return bio_tags
 
    for indices in start_end_indices:

        start_index = indices[ 0 ]
        end_index = indices[ 1 ]
        bio_tags[ start_index:end_index + 1 ] =   tensor([ config.bio_dict['B'] ] + [ config.bio_dict['I'] ] * (end_index - start_index ))

    return bio_tags

class Review:
    def __init__( self, review_id, text, aspect_term= None ):
        assert isinstance(review_id,str)
        assert isinstance(text, str)
        
        self.review_id = review_id
        self.text = text
        self.aspect_terms = aspect_term
        
        self.review_length = None
        self.tokenized_text = None # tokenized after the Dataset is parsed by the Vocab
        self.aspect_term_tokens = []
        self.tags = None
        self.aspect_positions = [] # populated after the Vocab is generated, contains a tuple of start index and end index
    
    def __str__(self):
        return str(self.__dict__)
    
    def set_tokenized_text(self, tokenized_text):
        self.tokenized_text = tokenized_text

    def set_aspect_term_positions(self, aspect_term_positions):
        self.aspect_positions = aspect_term_positions
    
    def set_aspect_term_tokens(self, aspect_term_tokens):
        self.aspect_term_tokens = aspect_term_tokens
    
    def set_tags(self, tags):
        self.tags = tags
    
    def set_review_length(self, length):
        self.review_length = length

class ReviewDataset(Dataset):
    def __init__( self, dataset_path, device= config.device, preprocessed= False, vocab= None ):
    
        self.device = device
        self.max_review_length = -1

        if not preprocessed:
            if isinstance(dataset_path, str):
                self.review_list = parse_xml(dataset_path)
            else:
                self.review_list = []
                for path in dataset_path:
                    self.review_list.append( parse_xml( path ) ) 
        else:
            self.review_list = []
            # when the dataset is already preprocessed and loaded into the file, just parse the strings and retreive the Record items
            print('reading file')
                    
            with open( dataset_path, 'r' ) as f:
                for line in f.readlines()[1:]:
                    line = line.strip().split('\t')
                    review_id = line.pop(0)
                    review_text = line.pop(0)
                    aspect_term = line if len( line ) > 0 else None          # remaining everything are the aspect terms
                    
                    self.review_list.append( Review( review_id, review_text, aspect_term ) )
            print('loading file complete')
        
        self.tokenizer = Vocab( self.review_list ) if vocab == None else vocab
        with open( './glove/mapping.json', 'w' ) as f:
                import json
                print('creating a mapping file')
                json.dump( self.tokenizer.get_vocab(), f )
        for review in self.review_list:
            
            review.tokenized_text = self.tokenizer.convert_text_to_sequence_numbers( review.text )
            
            if review.aspect_terms != None:
                aspect_terms_tokens = []
                aspect_terms_positions = []
                for aspect in review.aspect_terms:

                    aspect_term_tokens = self.tokenizer.convert_text_to_sequence_numbers( aspect, aspect= True )
                    aspect_term_positions = subfinder( review.tokenized_text, aspect_term_tokens )
                    if aspect_term_positions == None:
                        import sys
                        print(review, aspect_term_tokens, aspect_term_positions,self.tokenizer.convert_sequence_numbers_to_text(aspect_term_tokens),self.tokenizer.convert_sequence_numbers_to_text(review.tokenized_text))
                        sys.exit()
                    aspect_terms_tokens.append( aspect_term_tokens )
                    aspect_terms_positions.append( aspect_term_positions )

                
                review.set_aspect_term_tokens( aspect_terms_tokens ) 
                review.set_aspect_term_positions( aspect_terms_positions )
            
            padded_review, original_review_length = self.tokenizer.pad_sequence( review.tokenized_text, config.max_review_length )
            bio_tags = generate_bio_tags( review.aspect_positions, config.max_review_length )  

            review.set_tokenized_text( padded_review )
            review.set_review_length( original_review_length )
            review.set_tags( bio_tags )

    def write_to_file(self, filepath ):
        print('writing to file')
        with open( filepath, 'w' ) as f:
            f.write('review_id'+ config.sep +'review_text'+ config.sep +'aspect_terms\n')
            for review in self.review_list:
                f.write( review.review_id + config.sep + review.text + config.sep )
                if review.aspect_terms != None:
                    f.write( '\t'.join(review.aspect_terms) )
                f.write('\n')
        print("finished writing")
    
    def get_vocab(self):
        return self.tokenizer

    def get_review_list(self):
        return self.review_list

    def __len__(self):
        return len( self.review_list )

    def __getitem__(self, idx):

        data_item = self.review_list[idx]
        # padded_review, original_review_length = self.tokenizer.pad_sequence( data_item.tokenized_text, config.max_review_length )
        # bio_tags = generate_bio_tags( data_item.aspect_positions, config.max_review_length )
        
        item = {    
                    'review': data_item.tokenized_text,
                    'original_review_length': data_item.review_length,
                    'targets': data_item.tags
                }

        return item

class Vocab:

    def __init__( self, texts ):
        """
        :type texts: list of strings or Records
        :param texts: text of the reviews is either directly given or is extracted from the objects 
        Assumption: list contains elements of one kind alone
        """

        self.word_to_idx = {}
        self.index_to_word = {}
        self.size_of_vocab = 0
        
        # default unk token added in the beginning
        self.word_to_idx['<pad>'] = 0
        self.index_to_word[0] = '<pad>'
        self.size_of_vocab += 1

        self.word_to_idx['<unk>'] = 1
        self.index_to_word[1] = '<unk>'
        self.size_of_vocab += 1
        
        self.word_to_idx['<sos>'] = 2
        self.index_to_word[2] = '<sos>'
        self.size_of_vocab += 1
        
        self.word_to_idx['<eos>'] = 3
        self.index_to_word[ 3 ] = '<eos>'
        self.size_of_vocab += 1

        self.nlp = English()
        self.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)
        if isinstance( texts[0], str ):
            for doc in self.tokenizer.pipe( texts, batch_size= 100 ):
                for token in doc:
                    if not token.text in self.word_to_idx:
                        self.word_to_idx[ token.text ] = self.size_of_vocab 
                        self.index_to_word[ self.size_of_vocab ] = token.text
                        self.size_of_vocab += 1
                          
            
        elif isinstance( texts[0], Review ):
            for review in texts:
                tokens = self.tokenizer( review.text )
                for token in tokens:
                    if not token.text in self.word_to_idx:
                        self.word_to_idx[ token.text ] = self.size_of_vocab 
                        self.index_to_word[ self.size_of_vocab ] = token.text
                        self.size_of_vocab += 1
        
        else:
            raise Exception('input should be a list of stings or a list of Review objects')

    def print_vocab( self ):
        
        for idx, word in self.word_to_idx.items():
            print(idx, word) 
    
    def get_vocab(self):
        return self.word_to_idx

    def get_vocab_size(self):

        return self.size_of_vocab

    def convert_text_to_sequence_numbers(self, reviews, aspect= False):
        """ 
        :type reviews: list of strings or a string
        :param reviews: review's text
    
        :rtype:  list(list(int)) 
        """    
        if isinstance( reviews, list ): 
            text_sequences = []
            num_tokens = 0
            num_unk_tokens = 0
            for review in reviews:
                # tokenize and then convert into sequence numbers
                review_sequences = []
                for token in self.tokenizer( review ):
                    # sequence numbers are generated only for those sentences that are present in the vocab
                    if token.text in self.word_to_idx:
                        num_tokens += 1
                        review_sequences.append( self.word_to_idx[ token.text ] )
                    else:
                        num_unk_tokens += 1
                        review_sequences.append( self.word_to_idx[ '<unk>' ] )
                if not aspect:
                    reveiw_sequences = [ self.word_to_idx['<sos>'] ] + review_sequences + [ self.word_to_idx['<eos>'] ]
                text_sequences.append( review_sequences )
            print('num tokens: ',num_tokens,' num unk tokens: ', num_unk_tokens, ' percentage: ', 100*(num_unk_tokens/num_tokens))
           
            return text_sequences
        
        elif isinstance(reviews, str):
            review_sequences = []
            for token in self.tokenizer( reviews ):
                # sequence numbers are generated only for those sentences that are present in the vocab
                if token.text in self.word_to_idx:
                    review_sequences.append( self.word_to_idx[ token.text ] )
                else:
                    review_sequences.append( self.word_to_idx[ '<unk>' ] )
            if not aspect:
                return [ self.word_to_idx['<sos>'] ] + review_sequences + [ self.word_to_idx['<eos>'] ]
            else:
                return review_sequences
    def convert_sequence_numbers_to_text( self, reviews_sequences ):
        """ Converts list of sequence numbers to text

        :type reviews_sequences: list(list(int)) OR list(int)
        :param reviews_sequences: when a batch then list(list(int)), when a single review list(int) 
    
        :rtype: list of strings
        """    
        if isinstance(reviews_sequences[0], list):
            reviews = []
            for review_sequence in reviews_sequences:    
                review = ' '.join( [ self.index_to_word[ idx ] for idx in review_sequence if idx != self.word_to_idx['<sos>'] or idx != self.word_to_idx['<eos>']  ] )
                reviews.append( review )
            return reviews
                    
        elif isinstance( reviews_sequences[0], int):
            review = ' '.join( [ self.index_to_word[ idx ] for idx in reviews_sequences if idx != self.word_to_idx['<sos>'] or idx != self.word_to_idx['<eos>']] )
            return review

    def has_word(self, word):

        return word in self.word_to_idx

    def pad_sequence(self, input_sequence, length, padding= 'post', pad_character='<pad>'):
        
        original_length = len( input_sequence )
        if len( input_sequence ) < length:
            if padding == 'post':
                input_sequence = input_sequence + [ self.word_to_idx['<pad>'] ] * ( length - len(input_sequence) )
            elif padding == 'pre':
                input_sequence = [ self.word_to_idx['<pad>'] ] * ( length - len( input_sequence ) ) + input_sequence 
        
        return tensor( input_sequence ), tensor( original_length, dtype= torch.int64)

    @classmethod
    def from_files( cls, file_list ):
        texts = []
        for file_name in file_list:
            print(file_name)
            texts += parse_xml( file_name )

        return cls(texts)

if __name__ == "__main__":

    # process the raw xml
    vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path] )
    
    dataset = ReviewDataset(config.dataset_path, vocab= vocab)
    dataset.write_to_file('./datasets/train_data.tsv')
    dataset = ReviewDataset(config.test_dataset_path,vocab= vocab)
    dataset.write_to_file('./datasets/test_data.tsv')

    dataloader = DataLoader(dataset, batch_size= 2, shuffle= True, num_workers= 1)

    # # testing
    for i,batch in enumerate(dataloader):
         print('i', i)
         pprint(batch)
         input()
    x = [0,1,2,1,2,0]
    y = [0,1,2,1,0,0]
    print(evaluation_metrics(y, x))
