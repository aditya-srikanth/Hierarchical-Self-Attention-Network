import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.functional as F 
import numpy as np 
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from data_utils import create_embedding_matrix

class AttentionAspectionExtraction(nn.Module):
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), **kwargs):
        super(AttentionAspectionExtraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained( create_embedding_matrix(vocab, self.embedding_dim) )
        self.output_dim = output_dim
        
        rnn_model = kwargs.get('rnn_model','gru')
        if rnn_model == 'gru':   
            self.encoder = nn.GRU(
                                    self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional',True),
                                    num_layers = kwargs.get('num_rnn_layers',1),
                                    dropout = kwargs.get('dropout',0)
                                )

        elif rnn_model == 'lstm':
            self.encoder = nn.LSTM(
                                    self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional',True),
                                    num_layers = kwargs.get('num_rnn_layers',1),
                                    dropout = kwargs.get('dropout',0)
                                )
        
        self.weight_m = nn.Parameter( torch.rand( self.hidden_dim * 2, self.hidden_dim * 2 ) )
        self.bias_m = nn.Parameter( torch.rand( 1 ) )
        
        self.w_r = nn.Linear( self.hidden_dim * 2, output_dim )

    def forward( self, inputs ):
        
        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']
        aspect_tokens = inputs[ 'aspect_tokens' ]
        aspect_length = inputs['original_aspect_length']

        review = self.embedding( review )
        aspect_tokens = self.embedding( aspect_tokens )

        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)
        aspect_tokens = pack_padded_sequence( aspect_tokens, aspect_length, batch_first= True, enforce_sorted= False)

        review_h, _ = self.encoder( review )
        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0.0 )

        alpha = torch.nn.functional.softmax( torch.tanh( torch.bmm( torch.matmul( review_h, self.weight_m ), torch.transpose(review_h, 1, 2)  ) + self.bias_m ) , dim= 1 )
        
        s_i = torch.bmm( alpha, review_h )
        
        x = self.w_r( s_i )
        x = nn.functional.softmax(x, dim= 2)

        return x