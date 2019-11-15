import torch 
import config
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd

from data_utils import *
from model import *

def match(outputs, labels):
    results = []
    for predictions,targets in zip(outputs,labels):
        review_match = True
        for i in range( len( predictions ) ):
            if targets[i] == config.PAD: # special padding value, ignore this
                continue

            elif targets[i] == config.bio_dict['B']: # B tag seen
                matched = True
                begin = i
                while   i < len( predictions ) and targets[i] != config.bio_dict['O'] and targets[i] != config.PAD and \
                        not ( i > begin and targets[i] == config.bio_dict['B'] ): # B tag not seen again
                    
                    if targets[i] == predictions[i]:
                        i += 1
                    elif targets[i] != predictions[i]:
                        matched= False
                        break
                
                review_match = review_match and matched
        results.append( int(review_match) )

    return pd.DataFrame({'predictions': results})

def get_correct_indices(model,loader):
    with torch.no_grad():
        model.eval()
        print('evaluating')
        for _,batch in enumerate( loader ):
            batch = { k : v.to( config.device ) for k,v in batch.items()  }

            targets = batch[ 'targets' ]
            targets = pack_padded_sequence( targets, batch['original_review_length'], batch_first= True, enforce_sorted= False )
            targets,_ = pad_packed_sequence( targets, batch_first= True, padding_value= config.PAD )
            
            mask = ( targets < config.PAD )

            batch['targets'] = targets * mask.long()

            mask = mask.unsqueeze(2).float()
            if config.use_crf:
                _, outputs = model( batch, mask= mask, get_predictions= True ) 
            else:
                outputs = model( batch, mask= mask)
                outputs = torch.argmax( outputs, dim= 1 )
        
        results = match(outputs, targets)
        return results


vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path], store= config.mapping_file )
train_dataset = ReviewDataset(config.dataset_path, preprocessed= False, vocab= vocab)
test_dataset = ReviewDataset(config.test_dataset_path, preprocessed= False, vocab= vocab)
dataset = ConcatDataset([train_dataset,test_dataset])
loader = DataLoader( dataset, batch_size= len( dataset ), shuffle= False, num_workers= config.num_dataset_workers)


f_model = FusionAttentionAspectExtractionV2( vocab, embedding_path= config.word_embedding_path, use_crf= config.use_crf )
f_model.load_state_dict(torch.load('./model_weights/'+config.dataset+'_fusionv2_'+config.embedding+'.pt')) 
f_model = f_model.to(config.device)
print(f_model)

a_model = GlobalAttentionAspectExtraction( vocab, embedding_path= config.word_embedding_path, use_crf= config.use_crf )
a_model.load_state_dict(torch.load('./model_weights/'+config.dataset+'_attention_lstm_'+config.embedding+'.pt')) 
a_model = a_model.to(config.device)
print(a_model)

# b_model = BaseLineLSTM( vocab, embedding_path= config.word_embedding_path, use_crf= config.use_crf )
# b_model.load_state_dict(torch.load('./model_weights/'+config.dataset+'_lstm_'+config.embedding+'.pt')) 
# b_model = b_model.to(config.device)
# print(b_model)

# f_results = get_correct_indices(f_model, loader)
# a_results = get_correct_indices(a_model, loader)
# b_results = get_correct_indices(b_model, loader)

# a_correct = a_results & ~b_results 
# f_correct = f_results & ~a_results & ~b_results

# f_results.to_csv('./results/'+ config.dataset +'_fusionv2_predictions.csv', index= False)
# a_results.to_csv('./results/'+ config.dataset +'_attention_lstm_predictions.csv', index= False)
# b_results.to_csv('./results/'+ config.dataset +'_bilstm_predictions.csv', index= False)

# f_correct.to_csv('./results/'+ config.dataset +'_fusionv2_correct_predictions.csv', index= False)
# a_correct.to_csv('./results/'+ config.dataset +'_attention_lstm_correct_predictions.csv', index= False)

# print(f_results)
# print(a_results)
# print(b_results)
# print(a_correct)
# print(f_correct)


########## load pre generated results ############
f_results = pd.read_csv('./results/'+ config.dataset +'_fusionv2_predictions.csv')
a_results = pd.read_csv('./results/'+ config.dataset +'_attention_lstm_predictions.csv')
b_results = pd.read_csv('./results/'+ config.dataset +'_bilstm_predictions.csv')

a_correct = a_results & ~b_results 
f_correct = f_results & ~a_results & ~b_results

# print(f_correct)

f_correct = np.squeeze(f_correct.values)
a_correct = np.squeeze(a_correct.values)

f_sentences = np.squeeze(np.argwhere(f_correct == 1)).tolist()
a_sentences = np.squeeze(np.argwhere(a_correct == 1)).tolist()
# print(f_sentences)
# print(a_sentences)

dataset = train_dataset.get_review_list() + test_dataset.get_review_list()
print(len(dataset))
f_reviews = [ dataset[i] for i in list(f_sentences) ]
a_reviews = [ dataset[i] for i in list(a_sentences) ]

with open('./results/'+ config.dataset +'_fusion_sentences.txt','w') as f:
    
    with torch.no_grad():
        f_model.eval()
        print('generating scores')
        for _,batch in enumerate( loader ):
            batch = { k : v.to( config.device ) for k,v in batch.items()  }

            targets = batch[ 'targets' ]
            targets = pack_padded_sequence( targets, batch['original_review_length'], batch_first= True, enforce_sorted= False )
            targets,_ = pad_packed_sequence( targets, batch_first= True, padding_value= config.PAD )
            
            mask = ( targets < config.PAD )

            batch['targets'] = targets * mask.long()

            mask = mask.unsqueeze(2).float()
            if config.use_crf:
                _, outputs, global_context, beta = f_model( batch, mask= mask, get_predictions= True, yield_attention_weights= True ) 
            else:
                outputs, global_context, beta = f_model( batch, mask= mask, yield_attention_weights= True)
                outputs = torch.argmax( outputs, dim= 1 )
        
        betas = torch.stack([ beta[ i , :, : ] for i in f_sentences ])
        global_scores = torch.stack([ global_context[ i ] for i in f_sentences ])
        print(global_scores.shape)
        for i,g in enumerate(global_scores):
            print(i,g)
        input()
        torch.save(betas,'./model_weights/'+config.dataset+'_fusion_beta_scores.pt')
        torch.save(global_scores,'./model_weights/'+config.dataset+'_fusion_global_context_scores.pt')

        for review, global_score, beta in zip(f_reviews, global_scores, betas):
            f.write(review.text +' ####### '+ str(review.aspect_terms) + '\n')

        # print(global_context)



with open('./results/'+ config.dataset +'_attention_sentences.txt','w') as f:
    
    with torch.no_grad():
        a_model.eval()
        print('evaluating')
        for _,batch in enumerate( loader ):
            batch = { k : v.to( config.device ) for k,v in batch.items()  }

            targets = batch[ 'targets' ]
            targets = pack_padded_sequence( targets, batch['original_review_length'], batch_first= True, enforce_sorted= False )
            targets,_ = pad_packed_sequence( targets, batch_first= True, padding_value= config.PAD )
            
            mask = ( targets < config.PAD )

            batch['targets'] = targets * mask.long()

            mask = mask.unsqueeze(2).float()
            if config.use_crf:
                _, outputs, beta = a_model( batch, mask= mask, get_predictions= True, yield_attention_weights= True ) 
            else:
                outputs, beta = a_model( batch, mask= mask, yield_attention_weights= True)
                outputs = torch.argmax( outputs, dim= 1 )
        
        betas = torch.stack([ beta[ i , :, : ] for i in a_sentences ])
        torch.save(betas,'./model_weights/'+config.dataset+'_attention_lstm_beta_scores.pt')
        print(betas.shape)


    for review, beta in zip(a_reviews, betas):
            f.write(review.text + '########' + str(review.aspect_terms) + '\n')   


