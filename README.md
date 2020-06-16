# Relation_Classification_in_NLP

## Introduction
- Given 10 predefined relations like cause-effect, product-producer, etc, the goal was to define the relation and the direction of the relation b/w 2 entities in a sentence.
- I did this relation classification task in Python using Tensorflow library.
- I already had the pre-trained word embedding (glove) to train the data available. I had to setup Parts of Speech tag embedding and parse the shortest dependancy path b/w the 2 entities to build the model and then generate predictions. 


## Model Implementation
***Connectionist Bi-Directional RNN*** [1] - Representations of centre word along with succeeding & preceding words at any time step t.

- First I have defined two layers of GRU:

  - Forward Layer: This layer is used to calculate the forward representation of the sentence. Every forward layer returns the sequences of all previous layer representations. 

  - Backward Layer: In this GRU layer, to get backward representation, go_backwards flag is set True which reverses whatever sequence that you give to the GRU.

- Now, using the forward & backward layer, I have extracted forward & backward (reverse) representation of the sequence. 

- Then for every time step (i.e. index variable in model.py), I have extracted word representation from the forward & backward representations.

- After this, I have calculated the state at timestep t using the equation which is mentioned in [1] paper.

- After calculating the sequence representation for the whole batch, weighted sum of this final representation is done.

- This weighted sum is then passed to the decoder layer which basically classifies the each sentence (i.e. sequence) in the batch into 19 relations.

Figure of the model is included in the report.pdf under Model Architecture.

## References:
[1] Combining Recurrent and Convolutional Neural Networks for Relation Classification
