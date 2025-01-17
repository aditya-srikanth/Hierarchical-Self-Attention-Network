import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F 
import numpy as np 
from torchcrf import CRF

import config
from data_utils import create_embedding_matrix

class LSTM(nn.Module):
    
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(LSTM, self).__init__()
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
            self.crf = CRF( self.output_dim)
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

class AttentionAspectExtraction(nn.Module):
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(AttentionAspectExtraction, self).__init__()
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
            self.crf = CRF( self.output_dim)

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

    def forward( self, inputs, mask= None, get_predictions= False, yield_attention_weights= False ):
        
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
                if yield_attention_weights:
                    return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device), alpha 
                
                return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device)
            
            if yield_attention_weights:
                return - loss, alpha
            return - loss 

        x = nn.functional.log_softmax(x, dim= 2)
        if yield_attention_weights:
            return x * mask, alpha
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
        
        self.w_a = nn.Linear( self.hidden_dim * 2, self.hidden_dim * 2, bias= True ) # attention scores computation
        self.w_f = nn.Linear( self.hidden_dim * 4, self.hidden_dim * 2, bias= False )

        self.w_r = nn.Linear( self.hidden_dim * 2, output_dim )

        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF( self.output_dim)

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
    
    def forward( self, inputs, mask= None, get_predictions= False, yield_attention_weights=False ):
        
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

        # global_context_scores = torch.bmm( self.w_a( review_h ), final_hidden_state )
        global_context_scores = torch.bmm( self.w_a( review_h ), final_hidden_state ) + softmax_mask
        global_context_scores = F.softmax(global_context_scores, dim= 1).transpose(1,2)
        global_context = torch.bmm(global_context_scores, review_h)
        global_context = global_context.expand_as( review_h )
        
        review_h = torch.cat( [ review_h, global_context ], dim= 2 )
        
        review_h = self.w_f( review_h ) 
        
        x = self.w_r( review_h ).contiguous() 
        
        
        if self.use_crf:
            targets = inputs[ 'targets' ]
            
            mask = mask.squeeze_().type( torch.uint8 )
            loss = self.crf( x, targets, mask = mask )
            if get_predictions:
                temp = self.crf.decode( x )
                if yield_attention_weights:
                    return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device), global_context_scores
                return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device)
            if yield_attention_weights:
                return -loss, global_context_scores
            return - loss 

        x = nn.functional.log_softmax(x, dim= 2)
        if yield_attention_weights:
                return -loss, global_context_scores
        return x * mask
class Max_Margin_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def normalise(self, vector):
        denorm = torch.norm(vector, p=2, dim=-1)
        denorm = torch.max(torch.tensor(1e-6, dtype=torch.float).to('cuda'), denorm)
        return vector.div(denorm.unsqueeze(1))

    def forward(self, *_input, **kwargs):
       
        y_true, neg_samples, y_predicted = _input
      
        batch_size, _ = y_true.shape
        neg_sample_size, _ = neg_samples.shape
        #n_aspects, _ = aspect_embd.shape
        
        zs = self.normalise(y_true)
        
        ni = self.normalise(neg_samples)
        rs = self.normalise(y_predicted)
        
        ni = ni.repeat(batch_size, 1).view(batch_size, neg_sample_size, -1)
       
        zs_rs = torch.sum(torch.mul(zs, rs), dim=1, keepdims=True)
       
        rs_ni = torch.sum(torch.mul(rs.unsqueeze(1).repeat(1, neg_sample_size, 1),
                                    ni), dim=2, keepdims=True)
      
        loss = torch.sum(torch.max(torch.tensor(0, dtype=torch.float).to('cuda'),
                                   torch.tensor(1, dtype=torch.float).to('cuda') -
                                   zs_rs.unsqueeze(1).repeat(1, neg_sample_size, 1) +
                                   rs_ni), dim=1)
        
        loss = loss
        
        return torch.mean(loss)
class HSAN(nn.Module):
    
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, loss_fn=Max_Margin_Loss,lambda1=0, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(HSAN, self).__init__()
        self.loss_fn = loss_fn()
        self.lambda1 = lambda1
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

        self.w_a = nn.Linear( self.hidden_dim * 2, self.hidden_dim * 2, bias= True ) # attention scores computation
        self.w_f = nn.Linear( self.hidden_dim * 4, self.hidden_dim * 2, bias= False )
        
        self.weight_m = nn.Parameter( torch.Tensor( self.hidden_dim * 2, self.hidden_dim * 2 ) )
        # self.weight_m = nn.Parameter( torch.Tensor( self.hidden_dim * 4, self.hidden_dim * 4 ) )
        self.bias_m = nn.Parameter( torch.Tensor( 1 ) )
        
        self.w_r = nn.Linear( (self.hidden_dim * 2), output_dim, bias= False )
        
        self.latent      = nn.Linear( self.hidden_dim*2 , 20, bias= False )
        self.reconstruct = nn.Linear( 20,self.hidden_dim*2 , bias= False )
        # self.w_r = nn.Linear( self.hidden_dim * 4, output_dim, bias= True )
 
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

    def forward( self, inputs,inputs_neg=None, mask= None, get_predictions= False, yield_attention_weights= False ):
        
        if mask is None:
            mask = torch.ones((inputs['review'].shape[0], inputs['review'].shape[1], 1), dtype= torch.uint8, device=self.device) # shape ( batch_size, sequence_length, 1 )
        
        softmax_mask = (mask - 1) * 1e10

        review = inputs[ 'review' ]
        review_lengths = inputs['original_review_length'].to('cpu')

        review = self.embedding( review )
        
        
        if self.pos_dim != -1:
            review = torch.cat([review, inputs['pos_tags']], dim= 2)
        review = pack_padded_sequence(review, review_lengths, batch_first= True, enforce_sorted= False)
        
        
        if self.rnn_model == 'lstm':
            review_h, ( final_hidden_state, _) = self.encoder( review )
        elif self.rnn_model == 'gru':
            review_h, final_hidden_state = self.encoder( review )
            
        
            
        review_h, _         = pad_packed_sequence( review_h, batch_first= True, padding_value= 0 ) # shape: ( batch_size, seq_len, hidden_dim )
        
        
        #final_hidden_state     = torch.cat((final_hidden_state[-4,:,:], final_hidden_state[-3,:,:],final_hidden_state[-2,:,:],final_hidden_state[-1,:,:]), dim=1)
        #final_hidden_state_neg = torch.cat((final_hidden_state_neg[-4,:,:], final_hidden_state_neg[-3,:,:],final_hidden_state_neg[-2,:,:],final_hidden_state_neg[-1,:,:]), dim=1)
        
       
        
        
        
        ( batch_size, seq_len, hidden_dim ) = review_h.shape 
        final_hidden_state = final_hidden_state.view(config.num_layers, 2 if config.bidirectiional else 1, batch_size, hidden_dim // config.num_layers) # (num_layers, num_directions, batch, hidden_size)
        final_hidden_state = final_hidden_state.transpose(0,2) # (batch, num_directions, num_layers, hidden_size)
        final_hidden_state = final_hidden_state[:,:,-1,:].reshape(batch_size,-1,1) # (batch, num_directions * hidden_size, 1) 
        
        
        if get_predictions==False:
            review_neg = inputs_neg[ 'review' ]
            review_lengths_neg = inputs_neg['original_review_length'].to('cpu')
            review_neg = self.embedding( review_neg )
            
            review_neg = pack_padded_sequence(review_neg, review_lengths_neg, batch_first= True, enforce_sorted= False)
            
            if self.rnn_model == 'lstm':
                review_neg_h, ( final_hidden_state_neg, _) = self.encoder( review_neg )
            elif self.rnn_model == 'gru':
                review_neg_h, final_hidden_state_neg = self.encoder( review_neg )
            
            
            
            review_neg_h, _        = pad_packed_sequence( review_neg_h, batch_first= True, padding_value= 0 )
            final_hidden_state_neg = final_hidden_state_neg.view(config.num_layers, 2 if config.bidirectiional else 1, 20, hidden_dim // config.num_layers) # (num_layers, num_directions, batch, hidden_size)
            final_hidden_state_neg = final_hidden_state_neg.transpose(0,2) # (batch, num_directions, num_layers, hidden_size)
            final_hidden_state_neg = final_hidden_state_neg[:,:,-1,:].reshape(20,-1,1) # (batch, num_directions * hidden_size, 1)
            orig_sent              = final_hidden_state.squeeze(2)
            neg_sent =final_hidden_state_neg.squeeze(2)
            #print(orig_sent.shape,neg_sent.shape)
            latent       = self.latent( orig_sent)
            reconst_embd = self.reconstruct(latent)
            unsupervised_loss                                                                  = self.loss_fn( 
                                                                                                               orig_sent,
                                                                                                                neg_sent,
                                                                                                                reconst_embd
                                                                                                             )  
        
            
        

        # global_context_scores = torch.bmm( self.w_a( review_h ), final_hidden_state )
        global_context_scores = torch.bmm( self.w_a( review_h ), final_hidden_state ) + softmax_mask
        global_context_scores = F.softmax(global_context_scores, dim= 1).transpose(1,2)
        global_context = torch.bmm(global_context_scores, review_h)
        global_context = global_context.expand_as( review_h )
        
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
                if yield_attention_weights:
                    return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device), global_context_scores, alpha
                return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device)
            if yield_attention_weights:
                total_loss = -loss+self.lambda1*unsupervised_loss
                return - total_loss, global_context_scores, alpha
            total_loss = -loss+self.lambda1*unsupervised_loss
            return  total_loss

        x = nn.functional.log_softmax(x, dim= 2)
        if yield_attention_weights:
            return x * mask, global_context_scores, alpha
        return x * mask

class DECNN(nn.Module):
    def __init__(self, vocab, embedding_dim= config.word_embeding_dim, hidden_dim= config.hidden_dim, output_dim= len( config.bio_dict ), pos_dim= -1, use_crf= False, **kwargs):
        super(DECNN, self).__init__()
    
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding( len( vocab ), self.embedding_dim )
        self.output_dim = output_dim
        self.pos_dim = pos_dim        
        self.device = kwargs.get( 'device', config.device )
        self.vocab = vocab

        self.conv1=torch.nn.Conv1d( self.embedding_dim, 128, 5, padding=2 )
        self.conv2=torch.nn.Conv1d( self.embedding_dim, 128, 3, padding=1 )
        self.dropout=torch.nn.Dropout(config.dropout)

        self.conv3=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5=torch.nn.Conv1d(256, 256, 5, padding=2)
        
        self.linear_ae=torch.nn.Linear(256, self.output_dim)

        self.use_crf = use_crf

        if self.use_crf:
            self.crf = CRF( self.output_dim)

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
        review, _ = pad_packed_sequence( review, batch_first= True, padding_value= 0 )

        review = self.dropout(review).transpose(1,2)
        x_conv=torch.nn.functional.relu(torch.cat((self.conv1(review), self.conv2(review)), dim=1) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv3(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv4(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv5(x_conv) )
        x_conv=x_conv.transpose(1, 2)
        x_logit=self.linear_ae(x_conv)
        
        if self.use_crf:
            targets = inputs[ 'targets' ]
            
            mask = mask.squeeze_().type( torch.uint8 )
            loss = self.crf( x_logit, targets, mask = mask )
            if get_predictions:
                temp = self.crf.decode( x_logit )
                return - loss, torch.tensor( np.array( temp ) , dtype= torch.long, device= config.device)            
            
            return - loss
        
        x = nn.functional.log_softmax(x_logit, dim= 2)
        return x * mask