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
from data_utils import ReviewDataset, Vocab, Review, subfinder, generate_bio_tags, create_embedding_matrix, compute_accuracy
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
            for i,batch in enumerate( self.train_dataloader ):
                self.model.train()
                self.optimizer.zero_grad()
                
                batch['original_review_length'], perm_idx = batch['original_review_length'].sort( 0, descending= True )
                batch[ 'review' ] = batch[ 'review' ][ perm_idx ]
                batch[ 'bio_tags' ] = batch[ 'bio_tags' ][ perm_idx ]
                batch[ 'aspect_tokens' ] = batch[ 'aspect_tokens' ][ perm_idx ]
                batch['original_aspect_length'] = batch['original_aspect_length'][ perm_idx ]
                
                targets = batch[ 'bio_tags' ].to( self.device )
                targets = pack_padded_sequence(targets, batch['original_review_length'], batch_first= True)
                targets,_ = pad_packed_sequence(targets,batch_first=True)
                
                batch[ 'review' ] = batch[ 'review' ].to( self.device )
                batch['original_review_length'] = batch['original_review_length'].to( self.device )
                batch[ 'aspect_tokens' ] = batch[ 'aspect_tokens' ].to( self.device )
                batch['original_aspect_length'] = batch['original_aspect_length'].to( self.device )

                outputs = self.model( batch )

                loss = self.loss_function( outputs, targets )
                loss.backward()
                optimizer.step()
            
            print('loss ',loss, 'trainstep ', epoch)
            current_best = max( self.evaluate(current_best= current_best,path_save_best_model= model_save_path ), current_best )

    def evaluate(self, current_best= None, path_save_best_model= None):            
        with torch.no_grad():
            self.model.eval()
            for i,batch in enumerate( self.test_dataloader ):
                
                batch['original_review_length'], perm_idx = batch['original_review_length'].sort( 0, descending= True )
                batch[ 'review' ] = batch[ 'review' ][ perm_idx ]
                batch[ 'bio_tags' ] = batch[ 'bio_tags' ][ perm_idx ]
                batch[ 'aspect_tokens' ] = batch[ 'aspect_tokens' ][ perm_idx ]
                batch['original_aspect_length'] = batch['original_aspect_length'][ perm_idx ]
                
                targets = batch[ 'bio_tags' ].to( self.device )
                batch[ 'review' ] = batch[ 'review' ].to( self.device )
                batch['original_review_length'] = batch['original_review_length'].to( self.device )
                batch[ 'aspect_tokens' ] = batch[ 'aspect_tokens' ].to( self.device )
                batch['original_aspect_length'] = batch['original_aspect_length'].to( self.device )
                print(targets.shape)
                outputs = self.model( batch )
                outputs = torch.argmax( outputs, dim= 2 ).to('cpu')
                targets = torch.argmax( targets, dim= 2 ).to('cpu')

                # precision, recall, f_score, support = precision_recall_fscore_support( targets, outputs )
                accuracy = 0.0
                precision, recall, f_score = 0.0, 0.0, 0.0
                for row in zip(outputs, targets):
                    accuracy += compute_accuracy(row[1], row[0])

                print(' accuracy ', accuracy / outputs.shape[0] *100)
                input() 
                # print('precision ', precision/outputs.shape[0], ' recall ', recall/outputs.shape[0], ' f-score ', f_score/outputs.shape[0], ' accuracy ', accuracy*100 )

                
                if path_save_best_model != None: # implicit assumption: if this is given then you want to save the model
                    if current_best == None or current_best < accuracy:
                        print('saving model with accuracy: ', accuracy / outputs.shape[0] *100)
                        torch.save( self.model.state_dict(), path_save_best_model )

            return accuracy

if __name__ == "__main__":


    vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path] )
    
    train_dataset = ReviewDataset('./datasets/train_data.tsv',preprocessed= True, vocab= vocab)
    test_dataset = ReviewDataset('./datasets/test_data.tsv',preprocessed= True, vocab= vocab)

    model = AttentionAspectionExtraction( vocab, embedding_path= config.word_embedding_path )
    
    for p in model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / (p.shape[0]**0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    trainer = Trainer(model, train_dataset, test_dataset, loss_function, optimizer )
    trainer.run(1000, config.model_save_path )