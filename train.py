import sys
import os
import torch
from torch import tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from data_utils import ReviewDataset, Vocab, Review, subfinder, generate_bio_tags, create_embedding_matrix
from model import AttentionAspectionExtraction


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, loss_function, optimizer):
        
        self.train_dataset = train_dataset 
        self.test_dataset = test_dataset
        self.model = model
        self.loss_function = loss_function 
        self.optimizer = optimizer

        self.train_dataloader = DataLoader(train_dataset, batch_size= config.batch_size, shuffle= True, num_workers= 4)
        self.test_dataloader = DataLoader( test_dataset, batch_size= len(test_dataset), shuffle= False, num_workers= 4)
        self.device = torch.device( config.device if torch.cuda.is_available() else 'cpu')
        self.model.to( self.device )
        print('using device: ',self.device)

    def run(self, num_epochs, model_save_path ):

        current_best = -1
        for epoch in range( num_epochs ):
            
            self.model.train()
            for i,batch in enumerate( self.train_dataloader ):
            
                self.optimizer.zero_grad()
                
                batch = { k : v.to( self.device ) for k,v in batch.items() }
                
                targets = batch[ 'targets' ]
                targets = pack_padded_sequence(targets, batch['original_review_length'], batch_first= True, enforce_sorted= False)
                targets,_ = pad_packed_sequence(targets,batch_first=True,padding_value= 3.0)
                
                mask = ( targets < 3.0 ).float()

                targets = targets * mask
                targets = torch.argmax( targets, dim= 2 ).view( -1 ) # concat every instance

                outputs = self.model( batch ) * mask
                outputs = outputs.view( -1, 3 )
                # print(outputs, targets)
                loss = self.loss_function( outputs, targets )
                loss.backward()
                # print(loss)
                # input()
                optimizer.step()
            
            print('loss ',loss, 'trainstep ', epoch)
            current_best = max( self.evaluate(current_best= current_best,path_save_best_model= model_save_path ), current_best )

    def evaluate(self, current_best= None, path_save_best_model= None):

        with torch.no_grad():
            self.model.eval()
            for i,batch in enumerate( self.test_dataloader ):
                
                batch = { k : v.to( self.device ) for k,v in batch.items() }

                targets = batch[ 'targets' ]
                targets = pack_padded_sequence( targets, batch['original_review_length'], batch_first= True, enforce_sorted= False )
                targets,_ = pad_packed_sequence( targets, batch_first= True, padding_value= 3.0 )
                
                mask = ( targets < 3.0 ).float()

                targets = targets * mask
                targets = torch.argmax( targets, dim= 2 ).view( -1 ).to( 'cpu' ) # concat every instance
                
                outputs = ( self.model( batch ) ) * mask
                outputs = outputs.view( -1, 3 )
                outputs = torch.argmax( outputs, dim= 1 ).to( 'cpu' )
                
                accuracy = accuracy_score( targets, outputs )
                
                print( ' accuracy ', accuracy * 100, precision_recall_fscore_support(targets, outputs) )
                # input() 
                
                if path_save_best_model != None: # implicit assumption: if this is given then you want to save the model
                    if current_best == None or current_best < accuracy:
                        print('saving model with accuracy: ', accuracy)
                        torch.save( self.model.state_dict(), path_save_best_model )
                
            return accuracy

if __name__ == "__main__":


    vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path] )
    
    train_dataset = ReviewDataset('./datasets/train_data.tsv', preprocessed= True, vocab= vocab)
    test_dataset = ReviewDataset('./datasets/test_data.tsv', preprocessed= True, vocab= vocab)

    model = AttentionAspectionExtraction( vocab, embedding_path= config.word_embedding_path )
    
    for p in model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    loss_function = nn.NLLLoss(weight=tensor([0.1,0.45,0.45]).to(config.device))
    optimizer = torch.optim.Adam(model.parameters())

    trainer = Trainer(model, train_dataset, test_dataset, loss_function, optimizer )
    trainer.run(1000, config.model_save_path )