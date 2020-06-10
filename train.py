import argparse
import os
import time
import tensorflow as tf
import traceback
from datetime import datetime
from datasets.datafeeder import DataFeeder
from hparams import hparams
from models.tacotron import Tacotron
from util.text import sequence_to_text
from util import audio, infolog

log = infolog.log


class ValueWindow():
  def __init__(self, window_size=100):
    self._window_size = window_size
    self._values = []

  def append(self, x):
    self._values = self._values[-(self._window_size - 1):] + [x]

  @property
  def sum(self):
    return sum(self._values)

  @property
  def count(self):
    return len(self._values)

  @property
  def average(self):
    return self.sum / max(1, self.count)

  def reset(self):
    self._values = []


# tensorflow训练过程中保存的相关数据
def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('linear_outputs', model.linear_outputs)
    tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('loss_mel', model.mel_loss)
    tf.summary.scalar('loss_linear', model.linear_loss)
    tf.summary.scalar('regularization_loss', model.regularization_loss)
    tf.summary.scalar('stop_token_loss', model.stop_token_loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def train(log_dir, args):
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, args.input)
  # 显示模型的路径信息
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading training data from: %s' % input_path)
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams)

  # 初始化模型
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = Tacotron(hparams)
    model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.linear_targets, feeder.stop_token_targets, global_step)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=1)

  # 开始训练
  with tf.Session() as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())
      feeder.start_in_session(sess)

      while not coord.should_stop():
        start_time = time.time()
        step, loss, opt = sess.run([global_step, model.loss, model.optimize])
        time_window.append(time.time() - start_time)
        loss_window.append(loss)
        message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (step, time_window.average, loss, loss_window.average)
        log(message, slack=(step % args.checkpoint_interval == 0))

        if step % args.summary_interval == 0:
          summary_writer.add_summary(sess.run(stats), step)

        # 每隔一定的训练步数生成检查点
        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)
          log('Saving audio and alignment...')
          input_seq, spectrogram, alignment = sess.run([model.inputs[0], model.linear_outputs[0], model.alignments[0]])
          waveform = audio.inv_spectrogram(spectrogram.T)
          # 合成样音
          audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
          time_string = datetime.now().strftime('%Y-%m-%d %H:%M')
          # 画Encoder-Decoder对齐图
          infolog.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
            info='%s,  %s, step=%d, loss=%.5f' % (args.model, time_string, step, loss))
          # 显示合成样音的文本
          log('Input: %s' % sequence_to_text(input_seq))

    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)


def main():
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 1.0
  session = tf.Session(config=config)
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default='E:\\SRP\\TS')
  parser.add_argument('--input', default='training\\train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name')
  parser.add_argument('--hparams', default='')
  parser.add_argument('--restore_step', type=bool, default=True)
  parser.add_argument('--summary_interval', type=int, default=1000, help='每隔多少步进行一次总结')
  parser.add_argument('--checkpoint_interval', type=int, default=5000, help='每隔多少步生成检查点')
  parser.add_argument('--slack_url')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow日志等级')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, args)


if __name__ == '__main__':
  main()
