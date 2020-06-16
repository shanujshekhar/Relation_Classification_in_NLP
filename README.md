# Relation_Classification_in_NLP

Model Implementation:
---------------------
Connectionist Bi-Directional RNN [1]

(a) First I have defined two layers of GRU:

(1) Forward Layer: This layer is used to calculate the forward representation of the sentence. Every forward layer returns the sequences of all previous layer representations. 

(2) Backward Layer: In this GRU layer, to get backward representation, go_backwards flag is set True which reverses whatever sequence that you give to the GRU.

(b) Now, using the forward & backward layer, I have extracted forward & backward (reverse) representation of the sequence. 

(c) Then for every time step (i.e. index variable in model.py), I have extracted word representation from the forward & backward representations.

(d) After this, I have calculated the state at timestep t using the equation which is mentioned in [1] paper.

(e) After calculating the sequence representation for the whole batch, weighted sum of this final representation is done.

(f) This weighted sum is then passed to the decoder layer which basically classifies the each sentence (i.e. sequence) in the batch into 19 relations.

Figure of the model is included in the report.pdf under Model Architecture.

References:
-----------
[1] Combining Recurrent and Convolutional Neural Networks for Relation Classification