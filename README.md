# Supervised Attention Aspect Term Extraction

This project implements Aspect Term Extraction models on SemEval dataset.

## Getting Started
Currently the following models are implemented:
1. BaseLineLSTM
2. AttentionAspectExtraction
3. GlobalAttentionAspectExtraction
4. FusionAttentionAspectExtraction
5. FusionAttentionAspectExtractionV2

### Prerequisites
numpy, pytorch, spacy, pytorch-crf

### Installing
1. clone the repository
2. download glove embeddings and paste them in embeddings/glove
3. paste domain embeddings in embedding/domain_embedding
4. run concat_embeddings.py to generate glove-domain concactenated embeddings


## Running the tests

To change the model configuration, dataset and hyperparams,  