# Distributional-Shift-RCS

## We will be using the smallest variants of two different pretrained encoders, ELECTRA-Small (https://github.com/google-research/electra) and Ettin-XXS (https://github.com/jhu-clsp/ettin-encoder-vs-decoder).  

## Procedure - Each subgroup will proceed with a method, that is, each method's validity will be tested on all the datasets using both models. 

## To make computations less expensive, each pretrained model can be fine-tuned once on all the datasets and the fine-tuned weights can be uploaded here for everyone to use. Almost all the methods don't require optimizing the entire model, but just a small set of params and they can most likely be implemented without a GPU. 

## Uncertainty Methods
### Temperature Scaling - scaling the logits before softmax using a learnable parameter, to optimize probability distribution.
### Deep Ensemble - aggregating the results from multiple models (Likely expensive).
### MC (Monte Carlo) Dropout - aggregating the predicted results from the same model with active dropout layers. 
### OOD Detection (Newly added) - training a separate classification head for predicting the shift while the rest of model weights remain frozen

