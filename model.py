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
from torchcrf import CRF
import copy 

import config
from data_utils import create_embedding_matrix

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.weight_m = nn.Parameter( torch.Tensor( self.input_dim, self.input_dim ) )
        self.bias_m = nn.Parameter( torch.Tensor( 1 ) )
        
        self.weight_init()

    def weight_init(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                stdv = 1. / (p.shape[0] ** 0.5)
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def forward(self, x, mask= None):
    
        """ 
        :type x: torch.Tensor
        :param x: shape: ( batch_size, sequence_length, hidden_dim )
    
        :type mask: torch.FloatTensor
        :param mask: shape: ( batch_size, sequence_length )
    
        :rtype: torch.Tensor ( batch_size, sequence_length, hidden_dim )
        """    
        print('x',x)
        alpha = torch.bmm( torch.matmul( x, self.weight_m ), torch.transpose(x, -2, -1)  ) + self.bias_m
        
        print('alpha', alpha)

        if mask is not None:
            mask.unsqueeze_(1)
            mask = ( mask - 1 ) * 10e10
            alpha = alpha + mask
        
        print('alpha', alpha)

        alpha = torch.nn.functional.softmax( alpha , dim= -1 ) 

        print('alpha', alpha)
        context_vector = torch.matmul( alpha, x )

        return context_vector

class AttentionAspectionExtraction(nn.Module):
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(AttentionAspectionExtraction, self).__init__()
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding( len( vocab ), self.embedding_dim )
        self.output_dim = output_dim
        self.pos_dim = pos_dim        
        self.device = kwargs.get( 'device', config.device )
        self.vocab = vocab
        print('pos_dim ',pos_dim)
        rnn_model = kwargs.get( 'rnn_model', config.rnn_model )
        if rnn_model == 'gru':   
            self.encoder = nn.GRU(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )

        elif rnn_model == 'lstm':
            self.encoder = nn.LSTM(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )
        
        self.weight_m = nn.Parameter( torch.Tensor( self.hidden_dim * 2, self.hidden_dim * 2 ) )
        self.bias_m = nn.Parameter( torch.Tensor( 1 ) )
        
        self.w_r = nn.Linear( self.hidden_dim * 2, output_dim )

        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF( self.output_dim, batch_first= True )

        self.weight_init()

        

    def weight_init(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                stdv = 1. / (p.shape[0] ** 0.5)
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        self.embedding.weight = nn.Parameter(create_embedding_matrix(self.vocab, self.embedding_dim, 
                                                        dataset_path= config.word_embedding_path,  
                                                        save_weight_path= config.embedding_save_path ))
    def forward( self, inputs, mask= None, get_predictions= False ):
        
        if mask is None:
            mask = torch.ones((inputs['review'].shape[0], inputs['review'].shape[1], 1), dtype= torch.uint8, device=self.device) # shape ( batch_size, sequence_length, 1 )
        
        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']

        review = self.embedding( review )
        if self.pos_dim != -1:
            review = torch.cat([review, inputs['pos_tags']], dim= 2)
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)

        review_h, _ = self.encoder( review )
        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )

        alpha = torch.bmm( torch.matmul( review_h, self.weight_m ), torch.transpose(review_h, 1, 2)  ) + self.bias_m
        
        alpha = torch.nn.functional.softmax( alpha , dim= 2 ) # ( batch_size, sequence_length, attention_scores )
        

        s_i = torch.bmm( alpha, review_h )
        
        x = self.w_r( s_i ).contiguous() 
        
        
        if self.use_crf:
            targets = inputs[ 'targets' ]
            
            mask = mask.squeeze_().type( torch.uint8 )
            loss = self.crf( x, targets, mask = mask )
            if get_predictions:
                temp = self.crf.decode( x )
                return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device)
            
            return - loss 

        x = nn.functional.log_softmax(x, dim= 2)
        return x * mask

class MultiHeadAttentionAspectionExtraction(nn.Module): 
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, num_heads= 1, use_crf= False, **kwargs):
        super(MultiHeadAttentionAspectionExtraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = kwargs.get('device',config.device)
        self.embedding = nn.Embedding( len(vocab), self.embedding_dim )
        self.output_dim = output_dim
        self.pos_dim = pos_dim
        self.vocab = vocab
        rnn_model = kwargs.get('rnn_model',config.rnn_model)
        if rnn_model == 'gru':   
            self.encoder = nn.GRU(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )

        elif rnn_model == 'lstm':
            self.encoder = nn.LSTM(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )
        
        self.use_crf = use_crf
        self.num_heads = num_heads

        self.weight_m = nn.Parameter( torch.rand( self.num_heads, self.hidden_dim * 2, self.hidden_dim * 2 ) )
        self.w_t = nn.Linear( self.num_heads, 1 )
        self.w_r = nn.Linear( self.hidden_dim * 2, output_dim )

        if self.use_crf:
            self.crf = CRF( self.output_dim, batch_first= True )

        self.weight_init()

    def weight_init(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                stdv = 1. / (p.shape[0] ** 0.5)
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)

        self.embedding.weight = nn.Parameter(create_embedding_matrix(self.vocab, self.embedding_dim, 
                                                        dataset_path= config.word_embedding_path,  
                                                        save_weight_path= config.embedding_save_path ))


    def forward( self, inputs, mask= None, get_predictions= False ):
        if mask is None:
            mask = torch.ones((inputs['review'].shape[0], inputs['review'].shape[1], 1), dtype= torch.uint8, device=self.device) # shape ( batch_size, sequence_length, 1 )

        softmax_mask = ( (mask - 1) * 10e10 ).squeeze() # ( batch_size, seq_len )
        softmax_mask.unsqueeze_(1) # ( batch_size, 1, seq_len )
        softmax_mask.unsqueeze_(2) # ( batch_size, 1, 1, seq_len )
        softmax_mask = softmax_mask.expand( [-1, self.num_heads, softmax_mask.shape[3], -1] ) # ( batch_size, num_heads, seq_len, seq_len ) 

        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']

        review = self.embedding( review )
        if self.pos_dim != -1:
            review = torch.cat([review, inputs['pos_tags']], dim= 2)
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)
      
        review_h, _ = self.encoder( review )
        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )
        review_h = torch.unsqueeze( review_h, 1 ) # ( batch_size, 1, seq_len, hidden_dim )
        alpha = torch.matmul( review_h, self.weight_m )  # ( batch_size, num_heads, seq_len, hidden_dim )
        alpha = torch.matmul( alpha, torch.transpose(review_h, 2, 3) ) # ( batch_size, num_heads, seq_len, seq_len )
        alpha = alpha + softmax_mask
        alpha = torch.nn.functional.softmax( alpha , dim= 3 ) # ( batch_size, num_heads, sequence_length, attention_scores )
        
        s_i = torch.matmul( alpha, review_h ) # ( batch_size, num_heads, seq_len, hidden_dim )
        s_i = torch.transpose( s_i, 1, 2 )
        s_i = torch.transpose( s_i, 2, 3 ) # ( batch_size, seq_len, hidden_dim, num_heads )

        s_i = self.w_t( s_i ).squeeze()

        x = self.w_r( s_i ).contiguous() 
        
        
        
        if self.use_crf:
            targets = inputs[ 'targets' ]
            targets = pack_padded_sequence(targets, inputs['original_review_length'], batch_first= True, enforce_sorted= False)
            targets, _ = pad_packed_sequence(targets,batch_first=True,padding_value= 0)
            
            mask = mask.squeeze_().type( torch.uint8 )
            loss = self.crf( x, targets, mask = mask )
            if get_predictions:
                temp = self.crf.decode( x )
                return - loss, torch.tensor( np.array(temp) , dtype= torch.long, device= config.device)
            
            return - loss 

        x = nn.functional.log_softmax(x, dim= 2)
        return x * mask

class BaseLineLSTM(nn.Module):
    
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(BaseLineLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = self.embedding = nn.Embedding( len(vocab), self.embedding_dim )
        self.vocab = vocab

        self.output_dim = output_dim
        self.pos_dim = pos_dim
        rnn_model = kwargs.get('rnn_model', config.rnn_model)
        if rnn_model == 'gru':   
            self.encoder = nn.GRU(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )

        elif rnn_model == 'lstm':
            self.encoder = nn.LSTM(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )
        
        self.w_r = nn.Linear( self.hidden_dim * 2, output_dim )

        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF( self.output_dim, batch_first= True )
        self.drouput_layer = torch.nn.Dropout(p=config.dropout, inplace=True)
        self.weight_init()

    def weight_init(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                stdv = 1. / (p.shape[0] ** 0.5)
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        self.embedding.weight = nn.Parameter(create_embedding_matrix(self.vocab, self.embedding_dim, 
                                                        dataset_path= config.word_embedding_path,  
                                                        save_weight_path= config.embedding_save_path ))

    def forward( self, inputs, mask= None, get_predictions= False ):
        
        if mask is None:
            mask = torch.ones((inputs['review'].shape[0], inputs['review'].shape[1], 1), dtype= torch.uint8, device=self.device) # shape ( batch_size, sequence_length, 1 )
        
        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']

        review = self.embedding( review )
        if self.pos_dim != -1:
            review = torch.cat([review, inputs['pos_tags']], dim= 2)
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)

        review_h, _ = self.encoder( review )
        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )

        review_h = self.drouput_layer( review_h )

        x = self.w_r( review_h ).contiguous() 
        
        if self.use_crf:
            targets = inputs[ 'targets' ]
            
            mask = mask.squeeze_().type( torch.uint8 )
            loss = self.crf( x, targets, mask = mask )
            if get_predictions:
                temp = self.crf.decode( x )
                return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device)
            
            return - loss 

        x = nn.functional.log_softmax(x, dim= 2)
        return x * mask