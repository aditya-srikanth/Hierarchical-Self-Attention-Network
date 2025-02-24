import sys
import os
import torch
from torch import tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np 
import gc 
from tqdm import tqdm
import pprint

import config
from data_utils import ReviewDataset, Vocab, Review, subfinder, generate_bio_tags, create_embedding_matrix, evaluation_metrics
from model import AttentionAspectExtraction, LSTM, GlobalAttentionAspectExtraction,\
                HSAN, DECNN

class Trainer:
    def __init__(self, model, optimizer, train_dataset,test_dataset,num_folds= config.num_folds,loss_function= None):
        
        self.num_folds= num_folds
        assert num_folds >= 1

        
        self.use_crf = config.use_crf

        vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path], store= config.mapping_file )
    
        #self.train_dataset = ReviewDataset(config.dataset_path, preprocessed= False, vocab= vocab)
        #self.test_dataset = ReviewDataset(config.test_dataset_path, preprocessed= False, vocab= vocab)
    
        #self.model = model( vocab, embedding_path= config.word_embedding_path, use_crf= config.use_crf ).to(config.device)
        
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.model=model
        self.optimizer = optimizer(self.model.parameters())

        if not self.use_crf and loss_function is None:
            raise Exception(' Loss function must be specified when crf is not being used ')
        
        self.device = torch.device( config.device if torch.cuda.is_available() else 'cpu')
        self.model.to( self.device )
        
        
        print('using device: ',self.device)

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def _train(self, train_dataset, test_dataset, num_epochs, model_save_path= config.model_save_path, stats_save_path= config.save_stats_path ):

        train_dataloader = DataLoader( train_dataset, batch_size= config.batch_size, shuffle= True, num_workers= config.num_dataset_workers)
        test_dataloader = DataLoader( test_dataset, batch_size= len( test_dataset ), shuffle= False, num_workers= config.num_dataset_workers)
        neg_dataloader       = DataLoader(train_dataset,batch_size=20,shuffle= True, num_workers= config.num_dataset_workers)

        current_best = -1
        best_res = None
       
        for epoch in range( num_epochs ):

            avg_loss = 0.0
            self.model.train()

            print('\n*****************************************************************************************')
            for i,(batch,neg_batch) in enumerate(zip(train_dataloader,neg_dataloader)):

                self.optimizer.zero_grad()
                    
                batch     = { k : v.to( self.device ) for k,v in batch.items() }
                neg_batch = { k : v.to( self.device ) for k,v in neg_batch.items() }
                #print(batch)
                targets = batch[ 'targets' ]
                
                targets = pack_padded_sequence( targets, batch['original_review_length'].to('cpu'), batch_first= True, enforce_sorted= False)
                
                targets, _ = pad_packed_sequence( targets, batch_first=True, padding_value= config.PAD )
                
                mask = ( targets < config.PAD )
                targets = targets * mask.long()
                
                batch['targets'] = targets

                targets = targets.view( -1 ) # concat every instance
                
               
                outputs = self.model( batch,neg_batch, mask= mask.unsqueeze(2).float() )
                
                if self.use_crf:
                    #print("CRF")
                    loss = outputs
                    
                else:
                    #print("NO-CRF")
                    outputs = outputs.view( -1, 3 )
                    loss = self.loss_function( outputs, targets )
                #print(loss)
                loss.backward()
                self.optimizer.step()

            print('loss ',loss.item(), 'trainstep ', epoch)
            candidate_best, res = self.evaluate(test_dataloader, current_best= current_best,path_save_best_model= model_save_path )            
            
            # store the best result of the run
            if current_best < candidate_best:
                current_best = candidate_best
                best_res = res
            print('current best ', current_best)
            gc.collect()
        print('*****************************************************************************************\n')
        return current_best, best_res

    def evaluate(self, test_dataloader,current_best= None, path_save_best_model= None):
        
        with torch.no_grad():
            self.model.eval()
            print('evaluating')
            #neg_dataloader       = DataLoader(test_dataloader,batch_size=20,shuffle= True, num_workers= config.num_dataset_workers)
            for i,(batch) in enumerate(test_dataloader):
                
                batch      = { k : v.to( self.device ) for k,v in batch.items()  }
                #neg_batch  = { k : v.to( self.device ) for k,v in neg_batch.items()  }
                
                targets = batch[ 'targets' ]
                
                targets = pack_padded_sequence( targets, batch['original_review_length'].to('cpu'), batch_first= True, enforce_sorted= False )
                targets,_ = pad_packed_sequence( targets, batch_first= True, padding_value= config.PAD )
                
                mask = ( targets < config.PAD )

                batch['targets'] = targets * mask.long()

                targets = targets.view( -1 ) # concat every instance

                mask = mask.unsqueeze(2).float()
                if self.use_crf:
                    _, outputs = self.model( batch, mask= mask, get_predictions= True ) 
                    outputs = outputs.view(-1) # concatenate all predictions
                else:
                    outputs = self.model( batch,neg_batch, mask= mask)
                    outputs = outputs.view( -1, self.model.output_dim )
                    outputs = torch.argmax( outputs, dim= 1 )
                
                
                res = evaluation_metrics(outputs, targets)

                f_score = res['f_1']
                print( res )
                
                if path_save_best_model != None: # implicit assumption: if this is given then you want to save the model
                    if current_best == None or current_best < f_score:
                        print('saving model with f score: ', f_score)
                        torch.save( self.model.state_dict(), path_save_best_model )
                            
            return f_score, res

    def run(self, num_epochs, model_save_path, stats_save_path = config.save_stats_path ):
        
        with open( stats_save_path,'w' ) as f:
            if self.num_folds == 1:
               
                self._print_args()
                
                result = self._train(self.train_dataset, self.test_dataset, num_epochs, model_save_path, stats_save_path)
                headers = '\t'.join( [ str( header ) for header, _ in result[ 1 ].items() ]) + '\n'         
                string_results = '\t'.join( [ str( score ) for _, score in result[ 1 ].items() ]) + '\n'
                
                f.write(headers)
                f.write(string_results)
                f.flush()

            else:
                print("ELSE..")
                dataset = ConcatDataset( [ self.train_dataset, self.test_dataset ] )
                dataset_size = len( dataset )
                test_size = int( dataset_size * 0.2 ) 
                train_size = dataset_size - test_size
                lengths = [ train_size, test_size ] 

                for i in range( self.num_folds ):
                    print(' fold number: ',i)
                    self.model.weight_init()

                    splits = random_split( dataset, lengths )
                    train_dataset = splits[ 0 ]
                    test_dataset  = splits[ 1 ]

                    result = self._train( train_dataset, test_dataset, num_epochs, model_save_path, stats_save_path )


                    if i == 0:
                        headers = '\t'.join( [ str( header ) for header, _ in result[ 1 ].items() ]) + '\n'
                        f.write(headers)

                    string_results = '\t'.join( [ str( score ) for _, score in result[ 1 ].items() ]) + '\n'
                    f.write(string_results)
                    f.flush()

if __name__ == "__main__":
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    models = {
        'lstm': LSTM,
        'attention_lstm': AttentionAspectExtraction,
        'global_attention_lstm': GlobalAttentionAspectExtraction,
        'hsan': HSAN,
        'decnn' : DECNN
    }
    
    vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path], store= config.mapping_file )
    train_dataset = ReviewDataset(config.dataset_path, preprocessed= False, vocab= vocab)
    test_dataset = ReviewDataset(config.test_dataset_path, preprocessed= False, vocab= vocab)
    
    
    network =  models[ config.model ]( vocab, embedding_path= config.word_embedding_path, lambda1=config.lambda1,use_crf= config.use_crf ).to(config.device)
    trainer = Trainer( network, optimizers[ config.optimizer ],train_dataset,test_dataset,config.num_folds, loss_function= nn.NLLLoss() )
    trainer.run( config.num_epochs, config.model_save_path )
