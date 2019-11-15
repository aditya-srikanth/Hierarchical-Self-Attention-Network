#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import config
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd

from data_utils import *
from model import *


# In[2]:


def get_predicted_aspect( outputs, labels ):
    batch_predicted_as_aspects = []
    for predictions,targets in zip(outputs, labels):
        review_aspects = []
        for i in range( len( predictions ) ):
            if targets[i] == config.PAD:
                continue
            
            if predictions[i] == config.bio_dict['B']:
                begin = i 

                while   i < len( predictions ) and predictions[i] != config.bio_dict['O'] and targets[i] != config.PAD and                         not ( i > begin and predictions[i] == config.bio_dict['B'] ): # B tag not seen again
                    i += 1 # skip indices till the end of the aspect
                
                end = i
                review_aspects.append( ( begin, end ) )
                i -= 1
        batch_predicted_as_aspects.append(review_aspects)

    return batch_predicted_as_aspects

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
                while   i < len( predictions ) and targets[i] != config.bio_dict['O'] and targets[i] != config.PAD and                         not ( i > begin and targets[i] == config.bio_dict['B'] ): # B tag not seen again
                    
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
        predicted_aspects = get_predicted_aspect(outputs, targets)
        return results, predicted_aspects


# In[3]:


vocab = Vocab.from_files( [config.dataset_path, config.test_dataset_path], store= config.mapping_file )
train_dataset = ReviewDataset(config.dataset_path, preprocessed= False, vocab= vocab)
test_dataset = ReviewDataset(config.test_dataset_path, preprocessed= False, vocab= vocab)
dataset = ConcatDataset([train_dataset,test_dataset])
loader = DataLoader( dataset, batch_size= len( dataset ), shuffle= False, num_workers= config.num_dataset_workers)


# In[4]:


f_model = FusionAttentionAspectExtractionV2( vocab, embedding_path= config.word_embedding_path, use_crf= config.use_crf )
f_model.load_state_dict(torch.load('./model_weights/'+config.dataset+'_fusionv2_'+config.embedding+'.pt')) 
f_model = f_model.to(config.device)
print(f_model)

a_model = AttentionAspectExtraction( vocab, embedding_path= config.word_embedding_path, use_crf= config.use_crf )
a_model.load_state_dict(torch.load('./model_weights/'+config.dataset+'_attention_lstm_'+config.embedding+'.pt')) 
a_model = a_model.to(config.device)
print(a_model)

g_model = GlobalAttentionAspectExtraction( vocab, embedding_path= config.word_embedding_path, use_crf= config.use_crf )
g_model.load_state_dict(torch.load('./model_weights/'+config.dataset+'_global_attention_lstm_'+config.embedding+'.pt')) 
g_model = a_model.to(config.device)
print(a_model)

b_model = BaseLineLSTM( vocab, embedding_path= config.word_embedding_path, use_crf= config.use_crf )
b_model.load_state_dict(torch.load('./model_weights/'+config.dataset+'_lstm_'+config.embedding+'.pt')) 
b_model = b_model.to(config.device)
print(b_model)


# In[5]:


f_results, f_predictions = get_correct_indices(f_model, loader)
a_results, a_predictions = get_correct_indices(a_model, loader)
b_results, b_predictions = get_correct_indices(b_model, loader)
g_results, g_predictions = get_correct_indices(g_model, loader)

# f_results.to_csv('./results/'+ config.dataset +'_fusionv2_predictions.csv', index= False)
# a_results.to_csv('./results/'+ config.dataset +'_attention_lstm_predictions.csv', index= False)
# b_results.to_csv('./results/'+ config.dataset +'_bilstm_predictions.csv', index= False)
# g_results.to_csv('./results/'+ config.dataset +'_global_attention_lstm_predictions.csv', index= False)


# In[6]:


# ########## load pre generated results ############
# f_results = pd.read_csv('./results/'+ config.dataset +'_fusionv2_predictions.csv')
# a_results = pd.read_csv('./results/'+ config.dataset +'_attention_lstm_predictions.csv')
# b_results = pd.read_csv('./results/'+ config.dataset +'_bilstm_predictions.csv')
# g_results = pd.read_csv('./results/'+ config.dataset +'_global_attention_lstm_predictions.csv')


# In[ ]:


dataset = train_dataset.get_review_list() + test_dataset.get_review_list()
print(len(dataset))


# In[ ]:


f_correct = f_results & ~a_results & ~b_results
a_correct = a_results & ~b_results 
g_correct = g_results & ~b_results

# f_correct.to_csv('./results/'+ config.dataset +'_fusionv2_correct_predictions.csv', index= False)
# a_correct.to_csv('./results/'+ config.dataset +'_attention_lstm_correct_predictions.csv', index= False)
# g_results.to_csv('./results/'+ config.dataset +'_global_attention_lstm_correct_predictions.csv', index= False)


f_correct = np.squeeze(f_correct.values)
a_correct = np.squeeze(a_correct.values)
g_correct = np.squeeze(g_correct.values)

f_sentences = np.squeeze(np.argwhere(f_correct == 1)).tolist()
a_sentences = np.squeeze(np.argwhere(a_correct == 1)).tolist()
g_sentences = np.squeeze(np.argwhere(g_correct == 1)).tolist()

f_reviews = [ dataset[i] for i in list(f_sentences) ]
f_correct_predictions = [ (f_predictions[i],a_predictions[i],g_predictions[i],b_predictions[i]) for i in list(f_sentences) ]

a_reviews = [ dataset[i] for i in list(a_sentences) ]
a_correct_predictions = [ a_predictions[i] for i in list(a_sentences) ]

g_reviews = [ dataset[i] for i in list(g_sentences) ]
g_correct_predictions = [ g_predictions[i] for i in list(g_sentences) ]


# In[ ]:


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
#         for i,g in enumerate(global_scores):
#             print(i,g)
#         input()
    torch.save(betas,'./model_weights/'+config.dataset+'_fusion_beta_scores.pt')
    torch.save(global_scores,'./model_weights/'+config.dataset+'_fusion_global_context_scores.pt')

# store Fusion model results
with open('./results/'+ config.dataset +'_fusion_sentences.txt','w') as f:
    for review, prediction  in zip(f_reviews, f_correct_predictions):
        review_text_tokens = review.text.split(' ')
        f_prediction = [review_text_tokens[i: j] for (i,j) in prediction[0] ]
        a_prediction = [review_text_tokens[i: j] for (i,j) in prediction[1] ]
        g_prediction = [review_text_tokens[i: j] for (i,j) in prediction[2] ]
        b_prediction = [review_text_tokens[i: j] for (i,j) in prediction[3] ]
        f.write(review.text +' ####### '+ str(review.aspect_terms) + ' ###### fusion pred: '+ str(f_predictions) + ' ###### TSA pred: '+ str(a_predictions) + ' ###### global att pred: '+ str(g_predictions) + ' ###### blstm pred:'+ str(b_predictions) + '\n')

#############################################################################################################################

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



# store token-specific attention results
with open('./results/'+ config.dataset +'_attention_sentences.txt','w') as f:
    for review, prediction  in zip(dataset, a_predictions):
            review_text_tokens = review.text.split(' ')
            print(prediction)
            predictions = [''.join(review_text_tokens[i: j]) for (i,j) in prediction ]
            f.write(review.text +' ####### '+ str(review.aspect_terms) + ' ###### '+ str(predictions) + '\n')
            
#############################################################################################################################

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

# store global attention results
with open('./results/'+ config.dataset +'_global_attention_sentences.txt','w') as f:
    for review, prediction  in zip(g_reviews, g_correct_predictions):
            review_text_tokens = review.text.split(' ')
            predictions = [''.join(review_text_tokens[i: j]) for (i,j) in prediction ]
            f.write(review.text +' ####### '+ str(review.aspect_terms) + ' ###### '+ str(predictions) + '\n')


# In[7]:



f_fail = np.squeeze(1 - f_results.values)
g_fail = np.squeeze(1 - g_results.values)

f_fail = np.squeeze(np.argwhere(f_fail == 1)).tolist()
g_fail = np.squeeze(np.argwhere(g_fail == 1)).tolist()

# print(f_fail)
# print(g_fail)

f_reviews = [ dataset[i] for i in list(f_fail) ]
f_fail_predictions = [ f_predictions[i] for i in list(f_fail) ]
g_reviews = [ dataset[i] for i in list(g_fail) ]



with open('./results/'+ config.dataset +'_fusion_fail_sentences.txt','w') as f:
    
#     with torch.no_grad():
#         f_model.eval()
#         print('generating scores')
#         for _,batch in enumerate( loader ):
#             batch = { k : v.to( config.device ) for k,v in batch.items()  }

#             targets = batch[ 'targets' ]
#             targets = pack_padded_sequence( targets, batch['original_review_length'], batch_first= True, enforce_sorted= False )
#             targets,_ = pad_packed_sequence( targets, batch_first= True, padding_value= config.PAD )
            
#             mask = ( targets < config.PAD )

#             batch['targets'] = targets * mask.long()

#             mask = mask.unsqueeze(2).float()
#             if config.use_crf:
#                 _, outputs, global_context, beta = f_model( batch, mask= mask, get_predictions= True, yield_attention_weights= True ) 
#             else:
#                 outputs, global_context, beta = f_model( batch, mask= mask, yield_attention_weights= True)
#                 outputs = torch.argmax( outputs, dim= 1 )
        
#         betas = torch.stack([ beta[ i , :, : ] for i in f_fail ])
#         global_scores = torch.stack([ global_context[ i ] for i in f_fail ])
#         print(global_scores.shape)
# #         for i,g in enumerate(global_scores):
# #             print(i,g)
# #         input()
#         torch.save(betas,'./model_weights/'+config.dataset+'_fusion_fail_beta_scores.pt')
#         torch.save(global_scores,'./model_weights/'+config.dataset+'_fusion_fail_global_context_scores.pt')

        for review, prediction in zip(f_reviews,f_fail_predictions):
            if len(review.aspect_terms) > 1 and max([len(aspect) for aspect in review.aspect_terms ]) > 1:
                review_text_tokens = review.text.split(' ')
                predictions = [' '.join(review_text_tokens[i: j]) for (i,j) in prediction ]
                f.write(review.text +' ####### '+ str(review.aspect_terms) + ' ###### '+ str(predictions) + '\n')

with open('./results/'+ config.dataset +'_global_attention_fail_sentences.txt','w') as f:
    
#     with torch.no_grad():
#         g_model.eval()
#         print('evaluating')
#         for _,batch in enumerate( loader ):
#             batch = { k : v.to( config.device ) for k,v in batch.items()  }

#             targets = batch[ 'targets' ]
#             targets = pack_padded_sequence( targets, batch['original_review_length'], batch_first= True, enforce_sorted= False )
#             targets,_ = pad_packed_sequence( targets, batch_first= True, padding_value= config.PAD )
            
#             mask = ( targets < config.PAD )

#             batch['targets'] = targets * mask.long()

#             mask = mask.unsqueeze(2).float()
#             if config.use_crf:
#                 _, outputs, beta = g_model( batch, mask= mask, get_predictions= True, yield_attention_weights= True ) 
#             else:
#                 outputs, beta = g_model( batch, mask= mask, yield_attention_weights= True)
#                 outputs = torch.argmax( outputs, dim= 1 )
        
#         betas = torch.stack([ beta[ i , :, : ] for i in g_fail ])
#         torch.save(betas,'./model_weights/'+config.dataset+'_global_attention_lstm_fail_beta_scores.pt')
#         print(betas.shape)


    for review, beta in zip(g_reviews, betas):
        if len(review.aspect_terms) > 1 and max([len(aspect) for aspect in review.aspect_terms ]) > 1:
            f.write(review.text + '\t##' + str(review.aspect_terms) + '\n')   


