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

import config
from data_utils import create_embedding_matrix

class AttentionAspectionExtraction(nn.Module):
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), use_crf= False, **kwargs):
        super(AttentionAspectionExtraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = kwargs.get('device',config.device)
        self.embedding = nn.Embedding.from_pretrained( 
                                                        create_embedding_matrix(vocab, self.embedding_dim, 
                                                        dataset_path= config.word_embedding_path,  
                                                        save_weight_path= config.embedding_save_path )
                                                    )
        self.output_dim = output_dim
        
        rnn_model = kwargs.get('rnn_model','lstm')
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

        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF( self.output_dim, batch_first= True )

        self.weight_init()

        

    def weight_init(self):
        for p in self.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_normal_(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def forward( self, inputs, mask= None, get_predictions= False ):
        
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[1], 1), dtype= torch.uint8, device=self.device) # shape ( batch_size, sequence_length, 1 )

        softmax_mask = (mask - 1) * 10e10
        softmax_mask = softmax_mask.transpose(1,2)
        softmax_mask = softmax_mask.to( self.device )
        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']

        review = self.embedding( review )
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)
      
        review_h, _ = self.encoder( review )
        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )

        alpha = torch.tanh( torch.bmm( torch.matmul( review_h, self.weight_m ), torch.transpose(review_h, 1, 2)  ) + self.bias_m )
        alpha = alpha + softmax_mask
        
        alpha = torch.nn.functional.softmax( alpha , dim= 2 ) # ( batch_size, sequence_length, attention_scores )
        
        s_i = torch.bmm( alpha, review_h )
        
        x = torch.tanh( self.w_r( s_i ) ).contiguous() 
        
        
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

class MultiHeadAttentionAspectionExtraction(nn.Module):
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), num_heads= 1, use_crf= False, **kwargs):
        super(MultiHeadAttentionAspectionExtraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = kwargs.get('device',config.device)
        self.embedding = nn.Embedding.from_pretrained( 
                                                        create_embedding_matrix(vocab, self.embedding_dim, 
                                                        dataset_path= config.word_embedding_path,  
                                                        save_weight_path= config.embedding_save_path )
                                                    )
        self.output_dim = output_dim
        
        rnn_model = kwargs.get('rnn_model','lstm')
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
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_normal_(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def forward( self, inputs, mask= None, get_predictions= False ):
        
        # if mask is None:
        #     mask = torch.ones((x.shape[0], x.shape[1], 1), dtype= torch.uint8, device=self.device) # shape ( batch_size, sequence_length, 1 )

        # softmax_mask = (mask - 1) * 10e10
        # softmax_mask = softmax_mask.transpose(1,2)
        # softmax_mask = softmax_mask.to( self.device )
        
        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']

        review = self.embedding( review )
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)
      
        review_h, _ = self.encoder( review )
        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )
        review_h = torch.unsqueeze( review_h, 1 ) # ( batch_size, 1, seq_len, hidden_dim )
        alpha = torch.matmul( review_h, self.weight_m )  # ( batch_size, num_heads, seq_len, hidden_dim )
        alpha = torch.matmul( alpha, torch.transpose(review_h, 2, 3) ) # ( batch_size, num_heads, seq_len, seq_len )
        alpha = torch.tanh( alpha )
        # alpha = alpha + softmax_mask
        
        alpha = torch.nn.functional.softmax( alpha , dim= 3 ) # ( batch_size, num_heads,sequence_length, attention_scores )
        
        s_i = torch.matmul( alpha, review_h ) # ( batch_size, num_heads, seq_len, hidden_dim )
        s_i = torch.transpose( s_i, 1, 2 )
        s_i = torch.transpose( s_i, 2, 3 ) # ( batch_size, seq_len, hidden_dim, num_heads )

        s_i = self.w_t( s_i ).squeeze()

        x = torch.tanh( self.w_r( s_i ) ).contiguous() 
        
        
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
    
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), **kwargs):
        super(BaseLineLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained( 
                                                        create_embedding_matrix(vocab, self.embedding_dim, 
                                                        dataset_path= config.word_embedding_path,  
                                                        save_weight_path= config.embedding_save_path )
                                                    )
        self.output_dim = output_dim
        
        rnn_model = kwargs.get('rnn_model','lstm')
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
        
        self.weight_init()

    def weight_init(self):
        for p in self.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def forward( self, inputs ):
        
        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']

        review = self.embedding( review )
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)

        review_h, _ = self.encoder( review )
        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0.0 )
        
        x = torch.tanh( self.w_r( review_h ) ).contiguous() 

        x = nn.functional.log_softmax(x, dim= 2)
        return x

