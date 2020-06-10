import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.ops import array_ops, math_ops, variable_scope


# 参考论文中混合注意力能量公式：energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
def _location_sensitive_score(W_query, W_fil, W_keys):

	dtype = W_query.dtype
	num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

	v_a = tf.get_variable('attention_variable', shape=[num_units], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
	b_a = tf.get_variable('attention_bias', shape=[num_units], dtype=dtype, initializer=tf.zeros_initializer())

	return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


class LocationSensitiveAttention(BahdanauAttention):
	def __init__(self,
			num_units,
			memory,
			smoothing=False,
			cumulate_weights=True,
			name='LocationSensitiveAttention'):

		super(LocationSensitiveAttention, self).__init__(
			num_units=num_units,
			memory=memory,
			memory_sequence_length=None,
			probability_fn=None,
			name=name)

		self.location_convolution = tf.layers.Conv1D(filters=32,
			kernel_size=(31, ), padding='same', use_bias=True,
			bias_initializer=tf.zeros_initializer(), name='location_features_convolution')
		# 位置层（用于计算位置特征）
		self.location_layer = tf.layers.Dense(units=num_units, use_bias=False, dtype=tf.float32, name='location_features_layer')
		self._cumulate = cumulate_weights

	def __call__(self, query, state):
		previous_alignments = state
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

			# 计算query（W_query）
			processed_query = self.query_layer(query) if self.query_layer else query
			processed_query = tf.expand_dims(processed_query, 1)

			expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
			# 计算位置特征（W_location）
			f = self.location_convolution(expanded_alignments)
			processed_location_features = self.location_layer(f)

			# 使用能量函数公式计算得分
			energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

		alignments = self._probability_fn(energy, previous_alignments)
		if self._cumulate:
			next_state = alignments + previous_alignments
		else:
			next_state = alignments

		return alignments, next_state


# 计算注意力上下文向量和对齐效果
def _compute_attention(attention_mechanism, cell_output, attention_state, attention_layer):
	alignments, next_attention_state = attention_mechanism(cell_output, state=attention_state)
	expanded_alignments = array_ops.expand_dims(alignments, 1)
	# 上下文向量
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])

	if attention_layer is not None:
		attention = attention_layer(array_ops.concat([cell_output, context], 1))
	else:
		attention = context

	return attention, alignments, next_attention_state
