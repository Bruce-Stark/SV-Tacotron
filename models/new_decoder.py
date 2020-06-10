import collections
import tensorflow as tf
from hparams import hparams as hp
from .attention import _compute_attention
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops, check_ops, tensor_array_ops
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


# 线性投影层
class FrameProjection:

  def __init__(self, shape=hp.num_mels, activation=None, scope=None):
    super(FrameProjection, self).__init__()

    self.shape = shape
    self.activation = activation
    self.scope = 'linear_projection' if scope is None else scope
    self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

  def __call__(self, inputs):
    with tf.variable_scope(self.scope):
      return self.dense(inputs)


# 停止投影层（使用sigmoid激活函数）
class StopProjection:

  def __init__(self, is_training, shape=1, activation=tf.nn.sigmoid, scope=None):
    super(StopProjection, self).__init__()

    self.is_training = is_training
    self.shape = shape
    self.activation = activation
    self.scope = 'stop_token_projection' if scope is None else scope

  def __call__(self, inputs):
    with tf.variable_scope(self.scope):
      output = tf.layers.dense(inputs, units=self.shape, activation=None, name='projection_{}'.format(self.scope))
      return output if self.is_training else self.activation(output)


# 储存解码器单元状态
class TacotronDecoderCellState(
  collections.namedtuple("TacotronDecoderCellState", ("cell_state", "attention", "time", "alignments", "alignment_history"))):

  def replace(self, **kwargs):
    return super(TacotronDecoderCellState, self)._replace(**kwargs)


# Tacotron2解码器单元
class TacotronDecoderWrapper(RNNCell):

  def __init__(self, is_training, attention_mechanism, rnn_cell, frame_projection, stop_projection):

    super(TacotronDecoderWrapper, self).__init__()
    # 初始化解码器
    self._training = is_training
    self._attention_mechanism = attention_mechanism
    self._cell = rnn_cell
    self._frame_projection = frame_projection
    self._stop_projection = stop_projection
    self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

  def _batch_size_checks(self, batch_size):
    return [check_ops.assert_equal(batch_size,
      self._attention_mechanism.batch_size)]

  @property
  def output_size(self):
    return self._frame_projection.shape

  def state_size(self):

    return TacotronDecoderCellState(
      cell_state=self._cell._cell.state_size,
      time=tensor_shape.TensorShape([]),
      attention=self._attention_layer_size,
      alignments=self._attention_mechanism.alignments_size,
      alignment_history=())

  # 初始状态
  def zero_state(self, batch_size, dtype):

    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      cell_state = self._cell.zero_state(batch_size, dtype)
      with ops.control_dependencies(
        self._batch_size_checks(batch_size)):
        cell_state = nest.map_structure(
          lambda s: array_ops.identity(s, name="checked_cell_state"),
          cell_state)
      return TacotronDecoderCellState(
        cell_state=cell_state,
        time=array_ops.zeros([], dtype=tf.int32),
        attention=rnn_cell_impl._zero_state_tensors(self._attention_layer_size, batch_size, dtype),
        alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
        alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,
        dynamic_size=True))

  def __call__(self, inputs, state):
    prenet_x = inputs
    drop_rate = 0.5 if self._training else 0.0
    with tf.variable_scope('decoder_prenet' or 'prenet'):
      for i, size in enumerate(hp.prenet_depths):
        dense = tf.layers.dense(prenet_x, units=size, activation=tf.nn.relu, name='dense_%d' % (i + 1))
        prenet_x = tf.layers.dropout(dense, rate=drop_rate, training=self._training, name='dropout_%d' % (i + 1))
    prenet_output = prenet_x

    # RNN单元输入：拼接上下文向量和prenet网络的输出
    rnn_input = tf.concat([prenet_output, state.attention], axis=-1)

    # 单向RNN层
    rnn_output, next_cell_state = self._cell(tf.layers.dense(rnn_input, hp.decoder_depth), state.cell_state)

    previous_alignments = state.alignments
    previous_alignment_history = state.alignment_history
    # 计算注意力上下文向量和对齐效果
    context_vector, alignments, cumulated_alignments = _compute_attention(self._attention_mechanism,
      rnn_output,
      previous_alignments,
      attention_layer=None)

    # 拼接上下文向量和RNN输出
    projections_input = tf.concat([rnn_output, context_vector], axis=-1)

    # 计算预测帧和停止符
    cell_outputs = self._frame_projection(projections_input)
    stop_tokens = self._stop_projection(projections_input)
    # 记录历史对齐信息
    alignment_history = previous_alignment_history.write(state.time, alignments)
    # 下一状态
    next_state = TacotronDecoderCellState(
      time=state.time + 1,
      cell_state=next_cell_state,
      attention=context_vector,
      alignments=cumulated_alignments,
      alignment_history=alignment_history)

    return (cell_outputs, stop_tokens), next_state


class CustomDecoderOutput(collections.namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
  pass


class CustomDecoder(decoder.Decoder):

  def __init__(self, cell, helper, initial_state, output_layer=None):
    rnn_cell_impl.assert_like_rnncell(type(cell), cell)

    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:

      output_shape_with_unknown_batch = nest.map_structure(
        lambda s: tensor_shape.TensorShape([None]).concatenate(s),
        size)
      layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
        output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return CustomDecoderOutput(
      rnn_output=self._rnn_output_size(),
      token_output=self._helper.token_output_size,
      sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):

    dtype = nest.flatten(self._initial_state)[0].dtype
    return CustomDecoderOutput(
      nest.map_structure(lambda _: dtype, self._rnn_output_size()),
      tf.float32,
      self._helper.sample_ids_dtype)

  def initialize(self, name=None):

    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):

    with ops.name_scope(name, "CustomDecoderStep", (time, inputs, state)):
      (cell_outputs, stop_token), cell_state = self._cell(inputs, state)

      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
        time=time, outputs=cell_outputs, state=cell_state)

      (finished, next_inputs, next_state) = self._helper.next_inputs(
        time=time,
        outputs=cell_outputs,
        state=cell_state,
        sample_ids=sample_ids,
        stop_token_preds=stop_token)

    outputs = CustomDecoderOutput(cell_outputs, stop_token, sample_ids)
    return (outputs, next_state, next_inputs, finished)