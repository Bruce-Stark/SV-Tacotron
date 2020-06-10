import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, ResidualWrapper
from util.text import symbols
from .attention import LocationSensitiveAttention
from .helpers import TacoTestHelper, TacoTrainingHelper
from .new_decoder import FrameProjection, StopProjection, TacotronDecoderWrapper, CustomDecoder


class Tacotron():
  def __init__(self, hparams):
    self._hparams = hparams

  def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None, stop_token_targets=None, global_step=None):

    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embed_depth = 512
      embedding_table = tf.get_variable(
        'embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)

      # Encoder编码器模块（prenet网络和cbhg网络）
      prenet_outputs = prenet(embedded_inputs, is_training, hp.prenet_depths)                       # prenet_depths = [256, 256]
      encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training, hp.encoder_depth)  # encoder_depth = 256

      # 位置敏感注意力机制（attention_depth = 128）
      attention_mechanism = LocationSensitiveAttention(hp.attention_depth, encoder_outputs)

      # 解码器RNN（两层残差门控循环单元，decoder_depth = 1024）
      multi_rnn_cell = MultiRNNCell([
          ResidualWrapper(GRUCell(hp.decoder_depth)),
          ResidualWrapper(GRUCell(hp.decoder_depth))
        ], state_is_tuple=True)

      # 帧投影层（80*5）
      frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step)

      # 停止层（包含停止符，5）
      stop_projection = StopProjection(is_training, shape=hp.outputs_per_step)

      # 解码器单元
      decoder_cell = TacotronDecoderWrapper(is_training, attention_mechanism, multi_rnn_cell, frame_projection, stop_projection)

      if is_training:  # 训练
        helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, hp.outputs_per_step, global_step)
      else:  # 使用停止符进行预测
        helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)

      # 解码器初始化状态
      decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      (decoder_outputs, stop_token_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
         CustomDecoder(decoder_cell, helper, decoder_init_state), maximum_iterations=hp.max_iters)  # 80*5

      # 调整梅尔数组大小：从 80*5 到 80
      mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hp.num_mels])
      stop_token_outputs = tf.reshape(stop_token_outputs, [batch_size, -1])

      # 后处理网络（postnet_depth = 512）
      post_outputs = post_cbhg(mel_outputs, hp.num_mels, is_training, hp.postnet_depth)
      linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)  # num_freq = 2049

      # 从最终解码器状态获得对齐情况
      alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.mel_outputs = mel_outputs
      self.linear_outputs = linear_outputs
      self.stop_token_outputs = stop_token_outputs
      self.alignments = alignments
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      self.stop_token_targets = stop_token_targets

  def add_loss(self):
    with tf.variable_scope('loss') as scope:
      hp = self._hparams
      # 计算梅尔Loss
      self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      # 计算线性Loss
      self.linear_loss = tf.reduce_mean(tf.abs(self.linear_targets - self.linear_outputs))
      self.stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.stop_token_targets, logits=self.stop_token_outputs))

      # 计算正则化Loss
      regularization_weight = 1e-6
      all_vars = tf.trainable_variables()
      self.regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
        if not('bias' in v.name or 'Bias' in v.name)]) * regularization_weight

      # 计算总Loss
      self.loss = self.mel_loss + self.linear_loss + self.stop_token_loss + self.regularization_loss

  def add_optimizer(self, global_step):
    with tf.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        warmup_steps = 4000.0
        step = tf.cast(global_step + 1, dtype=tf.float32)
        self.learning_rate = hp.initial_learning_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)


# 1维卷积层
def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=channels,
      kernel_size=kernel_size,
      activation=None,
      padding='same')
    batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
    return activation(batched)


# prenet网络
def prenet(inputs, is_training, layer_sizes, scope=None):
  x = inputs
  drop_rate = 0.5 if is_training else 0.0
  with tf.variable_scope(scope or 'prenet'):
    for i, size in enumerate(layer_sizes):
      dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i+1))
  return x


# 编码器部分的CBHG网络
def encoder_cbhg(inputs, input_lengths, is_training, depth):
  input_channels = inputs.get_shape()[2]
  return cbhg(
    inputs,
    input_lengths,
    is_training,
    scope='encoder_cbhg',
    K=16,
    projections=[128, input_channels],
    depth=depth)


# 后处理网络部分的CBHG网络
def post_cbhg(inputs, input_dim, is_training, depth):
  return cbhg(
    inputs,
    None,
    is_training,
    scope='post_cbhg',
    K=8,
    projections=[256, input_dim],
    depth=depth)


# 普遍CBHG网络
def cbhg(inputs, input_lengths, is_training, scope, K, projections, depth):
  with tf.variable_scope(scope):
    with tf.variable_scope('conv_bank'):
      conv_outputs = tf.concat(
        [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)],
        axis=-1
      )

    # 最大池层
    maxpool_output = tf.layers.max_pooling1d(
      conv_outputs,
      pool_size=2,
      strides=1,
      padding='same')

    # 投影层和残差连接
    proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
    proj2_output = conv1d(proj1_output, 3, projections[1], lambda _:_, is_training, 'proj_2')
    highway_input = proj2_output + inputs

    half_depth = depth // 2
    assert half_depth*2 == depth, 'encoder and postnet depths must be even.'

    if highway_input.shape[2] != half_depth:
      highway_input = tf.layers.dense(highway_input, half_depth)

    # 高速公路网络
    for i in range(4):
      highway_input = highwaynet(highway_input, 'highway_%d' % (i+1), half_depth)
    rnn_input = highway_input

    # 双向RNN网络
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      GRUCell(half_depth),
      GRUCell(half_depth),
      rnn_input,
      sequence_length=input_lengths,
      dtype=tf.float32)
    return tf.concat(outputs, axis=2)  # Concat forward and backward


# 高速公路网络
def highwaynet(inputs, scope, depth):
  with tf.variable_scope(scope):
    H = tf.layers.dense(
      inputs,
      units=depth,
      activation=tf.nn.relu,
      name='H')
    T = tf.layers.dense(
      inputs,
      units=depth,
      activation=tf.nn.sigmoid,
      name='T',
      bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)
