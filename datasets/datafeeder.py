import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from util.text import text_to_sequence
from util.infolog import log

_batches_per_group = 32
_pad = 0
_stop_token_pad = 1


class DataFeeder(threading.Thread):

  def __init__(self, coordinator, metadata_filename, hparams):
    super(DataFeeder, self).__init__()
    self._coord = coordinator
    self._hparams = hparams
    self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    self._offset = 0

    # 读取txt文件
    self._datadir = os.path.dirname(metadata_filename)
    with open(metadata_filename, encoding='utf-8') as f:
      self._metadata = [line.strip().split('|') for line in f]
      hours = sum((int(x[2]) for x in self._metadata)) * hparams.frame_shift_ms / (3600 * 1000)
      log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))

    self._placeholders = [
      tf.placeholder(tf.int32, [None, None], 'inputs'),
      tf.placeholder(tf.int32, [None], 'input_lengths'),
      tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
      tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets'),
      tf.placeholder(tf.float32, [None, None], 'stop_token_targets')
    ]

    queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.inputs, self.input_lengths, self.mel_targets, self.linear_targets, self.stop_token_targets = queue.dequeue()
    self.inputs.set_shape(self._placeholders[0].shape)
    self.input_lengths.set_shape(self._placeholders[1].shape)
    self.mel_targets.set_shape(self._placeholders[2].shape)
    self.linear_targets.set_shape(self._placeholders[3].shape)
    self.stop_token_targets.set_shape(self._placeholders[4].shape)

  def start_in_session(self, session):
    self._session = session
    self.start()

  def run(self):
    try:
      while not self._coord.should_stop():
        self._enqueue_next_group()
    except Exception as e:
      traceback.print_exc()
      self._coord.request_stop(e)

  def _enqueue_next_group(self):
    start = time.time()
    n = self._hparams.batch_size
    r = self._hparams.outputs_per_step
    examples = [self._get_next_example() for i in range(n * _batches_per_group)]
    examples.sort(key=lambda x: x[-1])
    batches = [examples[i:i+n] for i in range(0, len(examples), n)]
    random.shuffle(batches)

    log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
    for batch in batches:
      feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
      self._session.run(self._enqueue_op, feed_dict=feed_dict)

  def _get_next_example(self):
    if self._offset >= len(self._metadata):
      self._offset = 0
      random.shuffle(self._metadata)
    meta = self._metadata[self._offset]
    self._offset += 1

    text = meta[3]
    input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
    linear_target = np.load(os.path.join(self._datadir, meta[0]))
    mel_target = np.load(os.path.join(self._datadir, meta[1]))
    stop_token_target = np.asarray([0.] * len(mel_target))
    return (input_data, mel_target, linear_target, stop_token_target, len(linear_target))


def _prepare_batch(batch, outputs_per_step):
  random.shuffle(batch)
  inputs = _prepare_inputs([x[0] for x in batch])
  input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
  mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
  linear_targets = _prepare_targets([x[2] for x in batch], outputs_per_step)
  stop_token_targets = _prepare_stop_token_targets([x[3] for x in batch], outputs_per_step)
  return (inputs, input_lengths, mel_targets, linear_targets, stop_token_targets)


def _prepare_inputs(inputs):
  max_len = max((len(x) for x in inputs))
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
  max_len = max((len(t) for t in targets)) + 1
  return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _prepare_stop_token_targets(targets, alignment):
  max_len = max((len(t) for t in targets)) + 1
  return np.stack([_pad_stop_token_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
  return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
  return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _pad_stop_token_target(t, length):
  return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=_stop_token_pad)


def _round_up(x, multiple):
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder
