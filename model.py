import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

	def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
		super(MyBasicAttentiveBiGRU, self).__init__()

		self.num_classes = len(ID_TO_CLASS)

		self.decoder = layers.Dense(units=self.num_classes)
		self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
		self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

		### TODO(Students) START
		# ...
		forward_layer = layers.GRU(hidden_size, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, use_bias=True)
		backward_layer = layers.GRU(hidden_size, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, go_backwards=True, use_bias=True)

		self.biDirectional = layers.Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(embed_dim, hidden_size))
		

		### TODO(Students) END

	def attn(self, rnn_outputs):
		### TODO(Students) START
		# ...
		
		M = tf.math.tanh(rnn_outputs)
		alpha = tf.nn.softmax( tf.tensordot( M,  self.omegas, [[2], [0]]), axis=1 )
		output_perseq = tf.multiply( rnn_outputs, alpha )
		output = tf.math.tanh(tf.reduce_sum(output_perseq, axis=1))

		### TODO(Students) END

		return output

	def call(self, inputs, pos_inputs, training):
		word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
		pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

		### TODO(Students) START
		# ...
		masking = tf.cast(inputs!=0, tf.float32)

		# [word + pos + dep] or [word + pos (without dep)]
		# H = self.biDirectional(tf.concat([word_embed, pos_embed], axis=2), mask=masking)

		# [word + dep] or [word (without dep)]
		H = self.biDirectional(word_embed, mask=masking)

		h_star = self.attn(H)
		
		logits = self.decoder(h_star)
		### TODO(Students) END

		return {'logits': logits}


class MyAdvancedModel(models.Model):

	def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
		super(MyAdvancedModel, self).__init__()
		### TODO(Students) START
		# ...
		self.num_classes = len(ID_TO_CLASS)
		self.hidden_size = hidden_size

		self.decoder = layers.Dense(units=self.num_classes)
		self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
		
		self.forward_layer = layers.GRU(hidden_size, return_sequences=True, use_bias=True)
		self.backward_layer = layers.GRU(hidden_size, return_sequences=True, go_backwards=True, use_bias=True)

		self.H = tf.Variable(
		  tf.random.truncated_normal([hidden_size, hidden_size], stddev = 0.001))

		self.ho = tf.constant(0.0, dtype=tf.float32, shape=[hidden_size, 1])


		### TODO(Students END

	def call(self, inputs, pos_inputs, training):
		
		### TODO(Students) START
		# ...
		
		word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
		pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

		masking = tf.cast(inputs!=0, tf.float32)
		
		# word_embed = tf.concat([word_embed, pos_embed], axis=2)

		batch_rep = []

		f_layer = self.forward_layer(word_embed, mask = masking)
		b_layer = self.backward_layer(word_embed, mask = masking)

		for hft, hbt in zip(f_layer, b_layer):
			
			ht_1 = tf.matmul(self.H, self.ho)

			index = 0
			max_length = word_embed.shape.as_list()[1]

			sent_rep = []

			for index in range(0, max_length):

				forward_word = hft[index:index+1, :]
				backward_word = hbt[ max_length-(index+1):max_length-index, :]

				ht = tf.add(forward_word, backward_word)
				ht = tf.add(ht, tf.reshape(ht_1, [1, -1]))
				ht_1 = ht

				sent_rep.append(ht)

			batch_rep.append(tf.concat(sent_rep, axis=0))

		
		batch_rep = tf.reshape(batch_rep, [word_embed.shape[0], word_embed.shape[1], self.hidden_size] )

		batch_rep = tf.reduce_sum(batch_rep, axis=1)

		logits = self.decoder(batch_rep)

		return {'logits': logits}	

		### TODO(Students END
