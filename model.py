import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torchcrf import CRF
import copy 

import config
from data_utils import create_embedding_matrix

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
        self.embedding.weight.requires_grad = False

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

class GlobalAttentionAspectExtraction(nn.Module):
    
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(GlobalAttentionAspectExtraction, self).__init__()
    
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding( len( vocab ), self.embedding_dim )
        self.output_dim = output_dim
        self.pos_dim = pos_dim        
        self.device = kwargs.get( 'device', config.device )
        self.vocab = vocab
        print('pos_dim ',pos_dim)
        self.rnn_model = kwargs.get( 'rnn_model', config.rnn_model )
    
        if self.rnn_model == 'gru':   
            self.encoder = nn.GRU(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )

        elif self.rnn_model == 'lstm':
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

        if self.rnn_model == 'lstm':
            review_h, ( global_context, _) = self.encoder( review )
        elif self.rnn_model == 'gru':
            review_h, global_context = self.encoder( review )

        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )
        
        ( batch_size, seq_len, hidden_dim ) = review_h.shape
        global_context = global_context.view(config.num_layers, 2 if config.bidirectiional else 1, batch_size, hidden_dim // config.num_layers) # (num_layers, num_directions, batch, hidden_size)
        global_context = global_context.transpose(0,2) # (batch, num_directions, num_layers, hidden_size)
        global_context = global_context[:,:,-1,:].reshape(batch_size,-1,1) # (batch, num_directions * hidden_size, 1)

        alpha = torch.bmm( torch.matmul( review_h, self.weight_m ), global_context  ) + self.bias_m
        alpha = torch.nn.functional.softmax( alpha, dim= 1 ) # ( batch_size, seq_len, 1 )
        alpha = alpha.expand_as( review_h ) # ( batch_size, seq_len, hidden_dim )

        s_i = alpha * review_h 
        
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

class FusionAttentionAspectExtraction(nn.Module):
    
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(FusionAttentionAspectExtraction, self).__init__()
    
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding( len( vocab ), self.embedding_dim )
        self.output_dim = output_dim
        self.pos_dim = pos_dim        
        self.device = kwargs.get( 'device', config.device )
        self.vocab = vocab
        print('pos_dim ',pos_dim)
        self.rnn_model = kwargs.get( 'rnn_model', config.rnn_model )
    
        if self.rnn_model == 'gru':   
            self.encoder = nn.GRU(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )

        elif self.rnn_model == 'lstm':
            self.encoder = nn.LSTM(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )

        self.w_a = nn.Linear( self.hidden_dim * 2, 1 ) # attention scores computation
        self.w_f = nn.Linear( self.hidden_dim * 4, self.hidden_dim * 2, bias= False )
        
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
        self.embedding.weight.requires_grad = False

    def forward( self, inputs, mask= None, get_predictions= False ):
        
        if mask is None:
            mask = torch.ones((inputs['review'].shape[0], inputs['review'].shape[1], 1), dtype= torch.uint8, device=self.device) # shape ( batch_size, sequence_length, 1 )
        
        softmax_mask = (mask - 1) * 1e10

        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']

        review = self.embedding( review )
        if self.pos_dim != -1:
            review = torch.cat([review, inputs['pos_tags']], dim= 2)
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)

        if self.rnn_model == 'lstm':
            review_h, ( global_context, _) = self.encoder( review )
        elif self.rnn_model == 'gru':
            review_h, global_context = self.encoder( review )

        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )
        
        global_context_scores = self.w_a( review_h ) + softmax_mask
        global_context_scores = F.softmax(global_context_scores, dim= 1).transpose(1,2)
        global_context = torch.bmm(global_context_scores, review_h)
        global_context = global_context.expand_as(review_h)
        
        review_h = torch.cat( [ review_h, global_context ], dim= 2 )
        
        review_h = self.w_f( review_h )
        
        alpha = torch.bmm( torch.matmul( review_h, self.weight_m ), torch.transpose(review_h, 1, 2)  ) + self.bias_m
        alpha = alpha + softmax_mask.transpose(1,2)
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

class FusionAttentionAspectExtractionV2(nn.Module):
    
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(FusionAttentionAspectExtractionV2, self).__init__()
    
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding( len( vocab ), self.embedding_dim )
        self.output_dim = output_dim
        self.pos_dim = pos_dim        
        self.device = kwargs.get( 'device', config.device )
        self.vocab = vocab
        print('pos_dim ',pos_dim)
        self.rnn_model = kwargs.get( 'rnn_model', config.rnn_model )
    
        if self.rnn_model == 'gru':   
            self.encoder = nn.GRU(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )

        elif self.rnn_model == 'lstm':
            self.encoder = nn.LSTM(
                                    self.embedding_dim + self.pos_dim if self.pos_dim != -1 else self.embedding_dim,
                                    self.hidden_dim,
                                    bidirectional= kwargs.get('bidirectional', config.bidirectiional),
                                    num_layers = kwargs.get('num_rnn_layers', config.num_layers),
                                    dropout = kwargs.get('dropout', config.dropout)
                                )

        self.w_a = nn.Linear( self.hidden_dim * 2, self.hidden_dim * 2, bias= False ) # attention scores computation
        # self.w_f = nn.Linear( self.hidden_dim * 4, self.hidden_dim * 2, bias= False )
        
        # self.weight_m = nn.Parameter( torch.Tensor( self.hidden_dim * 2, self.hidden_dim * 2 ) )
        self.weight_m = nn.Parameter( torch.Tensor( self.hidden_dim * 4, self.hidden_dim * 4 ) )
        self.bias_m = nn.Parameter( torch.Tensor( 1 ) )
        
        # self.w_r = nn.Linear( self.hidden_dim * 2, output_dim, bias= False )
        self.w_r = nn.Linear( self.hidden_dim * 4, output_dim, bias= False )

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
        self.embedding.weight.requires_grad = False

    def forward( self, inputs, mask= None, get_predictions= False ):
        
        if mask is None:
            mask = torch.ones((inputs['review'].shape[0], inputs['review'].shape[1], 1), dtype= torch.uint8, device=self.device) # shape ( batch_size, sequence_length, 1 )
        
        softmax_mask = (mask - 1) * 1e10

        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length']

        review = self.embedding( review )
        if self.pos_dim != -1:
            review = torch.cat([review, inputs['pos_tags']], dim= 2)
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)

        if self.rnn_model == 'lstm':
            review_h, ( final_hidden_state, _) = self.encoder( review )
        elif self.rnn_model == 'gru':
            review_h, final_hidden_state = self.encoder( review )

        review_h, _ = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )
        
        ( batch_size, seq_len, hidden_dim ) = review_h.shape
        final_hidden_state = final_hidden_state.view(config.num_layers, 2 if config.bidirectiional else 1, batch_size, hidden_dim // config.num_layers) # (num_layers, num_directions, batch, hidden_size)
        final_hidden_state = final_hidden_state.transpose(0,2) # (batch, num_directions, num_layers, hidden_size)
        final_hidden_state = final_hidden_state[:,:,-1,:].reshape(batch_size,-1,1) # (batch, num_directions * hidden_size, 1)

        global_context_scores = torch.bmm( self.w_a( review_h ), final_hidden_state )
        global_context_scores = torch.bmm( self.w_a( review_h ), final_hidden_state ) + softmax_mask
        global_context_scores = F.softmax(global_context_scores, dim= 1).transpose(1,2)
        global_context = torch.bmm(global_context_scores, review_h)
        global_context = global_context.expand_as( review_h )
        
        review_h = torch.cat( [ review_h, global_context ], dim= 2 )
        
        # review_h = self.w_f( review_h )
        
        alpha = torch.bmm( torch.matmul( review_h, self.weight_m ), torch.transpose(review_h, 1, 2)  ) + self.bias_m
        alpha = alpha + softmax_mask.transpose(1,2)
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

