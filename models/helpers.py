import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper


class TacoTrainingHelper(Helper):
  def __init__(self, inputs, targets, output_dim, r, global_step):
    with tf.name_scope('TacoTrainingHelper'):
      self._batch_size = tf.shape(inputs)[0]
      self._output_dim = output_dim
      self._reduction_factor = r
      self._ratio = tf.convert_to_tensor(1.)
      self.global_step = global_step
      self._targets = targets[:, r-1::r, :]

      num_steps = tf.shape(self._targets)[1]
      self._lengths = tf.tile([num_steps], [self._batch_size])

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def token_output_size(self):
    return self._reduction_factor

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):
    self._ratio = _teacher_forcing_ratio_decay(1., self.global_step)
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

  def sample(self, time, outputs, state, name=None):
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def next_inputs(self, time, outputs, state, sample_ids, stop_token_preds, name=None):
    with tf.name_scope(name or 'TacoTrainingHelper'):
      finished = (time + 1 >= self._lengths)

      next_inputs = tf.cond(tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), self._ratio),
              lambda: self._targets[:, time, :], lambda: outputs[:, -self._output_dim:])

      return (finished, next_inputs, state)


class TacoTestHelper(Helper):
  def __init__(self, batch_size, output_dim, r):
    with tf.name_scope('TacoTestHelper'):
      self._batch_size = batch_size
      self._output_dim = output_dim
      self._reduction_factor = r

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def token_output_size(self):
    return self._reduction_factor

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

  def sample(self, time, outputs, state, name=None):
    return tf.tile([0], [self._batch_size])

  def next_inputs(self, time, outputs, state, sample_ids, stop_token_preds, name=None):

    with tf.name_scope('TacoTestHelper'):
      finished = tf.reduce_any(tf.cast(tf.round(stop_token_preds), tf.bool))
      next_inputs = outputs[:, -self._output_dim:]
      return (finished, next_inputs, state)


# 解码器 <GO> frame（0矩阵）
def _go_frames(batch_size, output_dim):
  return tf.tile([[0.0]], [batch_size, output_dim])


# teacher forcing ratio机制
def _teacher_forcing_ratio_decay(init_tfr, global_step):

  tfr = tf.train.cosine_decay(init_tfr,
          global_step = global_step - 20000,
          decay_steps = 200000,
          alpha = 0.,
          name = 'tfr_cosine_decay')

  narrow_tfr = tf.cond(
          tf.less(global_step, tf.convert_to_tensor(20000)),
          lambda: tf.convert_to_tensor(init_tfr),
          lambda: tfr)

  return narrow_tfr
