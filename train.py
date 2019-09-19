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

import config
from data_utils import ReviewDataset, Vocab, Review, subfinder, generate_bio_tags, create_embedding_matrix, evaluation_metrics
from model import AttentionAspectionExtraction, BaseLineLSTM, MultiHeadAttentionAspectionExtraction

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, loss_function, optimizer, num_folds = 1):
        
        self.train_dataset = train_dataset 
        self.test_dataset = test_dataset
        self.model = model
        self.loss_function = loss_function 
        self.optimizer = optimizer
        self.num_folds = num_folds
        self.use_crf = self.model.use_crf if hasattr(self.model, 'use_crf') else False
        
        self.device = torch.device( config.device if torch.cuda.is_available() else 'cpu')
        self.model.to( self.device )
        print('using device: ',self.device)

    def _train(self, train_dataset, test_dataset, num_epochs, model_save_path= config.model_save_path, stats_save_path= config.save_stats_path ):

        train_dataloader = DataLoader( train_dataset, batch_size= config.batch_size, shuffle= False, num_workers= config.num_dataset_workers)
        test_dataloader = DataLoader( test_dataset, batch_size= len( test_dataset ), shuffle= False, num_workers= config.num_dataset_workers)

        current_best = -1
        best_res = None

        for epoch in range( num_epochs ):

            avg_loss = 0.0
            self.model.train()

            print('*****************************************************************************************\n')
            for i,batch in enumerate( tqdm(train_dataloader) ):

                self.optimizer.zero_grad()
                    
                batch = { k : v.to( self.device ) for k,v in batch.items() }
                targets = batch[ 'targets' ]
                targets = pack_padded_sequence( targets, batch['original_review_length'], batch_first= True, enforce_sorted= False)
                targets, _ = pad_packed_sequence( targets, batch_first=True, padding_value= 3 )
                
                mask = ( targets < 3 )
                
                targets = targets * mask.long()
                targets = targets.view( -1 ) # concat every instance

                outputs = self.model( batch, mask= mask.unsqueeze(2).float() )

                if self.use_crf:
                    loss = outputs

                else:
                    outputs = outputs.view( -1, 3 )
                    loss = self.loss_function( outputs, targets )

                loss.backward()
                optimizer.step()

            print('loss ',loss.item(), 'trainstep ', epoch)
            candidate_best, res = self.evaluate(test_dataloader, current_best= current_best,path_save_best_model= model_save_path )            
            
            if current_best < candidate_best:
                current_best = candidate_best
                best_res = res
            # gc.collect()
        print('*****************************************************************************************\n')
        return current_best, best_res

    def evaluate(self, test_dataloader,current_best= None, path_save_best_model= None):
        
        with torch.no_grad():
            self.model.eval()
            print('evaluating')
            for _,batch in enumerate( test_dataloader ):
                batch = { k : v.to( self.device ) for k,v in batch.items()  }

                targets = batch[ 'targets' ]
                targets = pack_padded_sequence( targets, batch['original_review_length'], batch_first= True, enforce_sorted= False )
                targets,_ = pad_packed_sequence( targets, batch_first= True, padding_value= config.PAD )
                
                mask = ( targets < 3.0 )
                targets = targets.view( -1 ) # concat every instance

                mask = mask.unsqueeze(2).float()
                if self.use_crf:
                    _, outputs = self.model( batch, mask= mask, get_predictions= True ) 
                    outputs = outputs.view(-1) # concatenate all predictions
                else:
                    outputs = self.model( batch, mask= mask)
                    outputs = outputs.view( -1, 3 )
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

            if self.num_folds <= 1:
                result = self._train(self.train_dataset, self.test_dataset, num_epochs, model_save_path, stats_save_path)
                headers = '\t'.join( [ str( header ) for header, _ in result[ 1 ].items() ]) + '\n'         
                string_results = '\t'.join( [ str( score ) for _, score in result[ 1 ].items() ]) + '\n'
                
                f.write(headers)
                f.write(string_results)
                f.flush()

            elif self.num_folds > 1:
                dataset = ConcatDataset( [ self.train_dataset, self.test_dataset ] )
                dataset_size = len( dataset )
                fold_size = dataset_size // self.num_folds
                lengths = [ fold_size ] * self.num_folds # creates equal size folds 
                lengths.append( dataset_size - sum( lengths ) )
                splits = random_split( dataset, lengths )
                
                for i in range( self.num_folds ):
                    print(' fold number: ',i)
                    self.model.weight_init()

                    test_dataset = splits[ i ]
                    indices = [ j for j in range( self.num_folds ) if j != i ]
                    train_dataset = ConcatDataset( [ splits[ index ] for index in indices ] )
                    result = self._train( train_dataset, test_dataset, num_epochs, model_save_path, stats_save_path )
                

                    if i == 0:
                        headers = '\t'.join( [ str( header ) for header, _ in result[ 1 ].items() ]) + '\n'
                        f.write(headers)

                    string_results = '\t'.join( [ str( score ) for _, score in result[ 1 ].items() ]) + '\n'
                    f.write(string_results)
                    f.flush()
        
    
if __name__ == "__main__":


    vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path] )
    
    train_dataset = ReviewDataset('./datasets/train_data.tsv', preprocessed= True, vocab= vocab)
    test_dataset = ReviewDataset('./datasets/test_data.tsv', preprocessed= True, vocab= vocab)
    
    model = MultiHeadAttentionAspectionExtraction( vocab, embedding_path= config.word_embedding_path, num_heads= 2, use_crf= False )

    weight=tensor([ 0.2, 0.4, 0.4 ]).to( config.device )
    loss_function = nn.NLLLoss(weight= weight)
    # loss_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    trainer = Trainer(model, train_dataset, test_dataset, loss_function, optimizer, num_folds= 10 )
    trainer.run(config.num_epochs, config.model_save_path )
