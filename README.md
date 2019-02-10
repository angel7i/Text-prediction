# Text Prediction

After you have trained a neural network, you would want to save it for future use and predict new 
inputs, the new inputs need to be transformed in vectors that the model recognize, for this is
necessary load the vocabulary generated during the training.

## Prerequisites

*   [text_classification](https://github.com/google/eng-edu/tree/master/ml/guides/text_classification) -
This repository is the base example, contains samples to classify text using machine learning.

## Tests

We create tests to generate the outputs and then load them in another process.

1.  *integration_test* - Functions to train the models, modified to save the vectorizer information
that contains the vocabulary learned from the training dataset. 

2.  *predict_sequence_model_test* - With the model and the vectorizer created we can load them
on demand to predict new inputs, the output of this model generate five scores for each input, 
the major confidence means that the input is in that class.  

3.  *predict_fine_sequence_model_test* - This test evaluate the inputs of tuned sequence model, 
for this model we have 15 possibles categories, once loaded the model and the vectorizer we can 
pass new inputs and classify them in one category.
