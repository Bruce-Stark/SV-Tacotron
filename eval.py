import os
import re
import io
import argparse
import numpy as np
from util import audio
import tensorflow as tf
from xpinyin import Pinyin
from hparams import hparams
from models.tacotron import Tacotron
from util.text import text_to_sequence

sentence = ['中文的语音合成效果比英文的好']
# sentence = ['zhong1 wen2 de5 yu3 yin1 he2 cheng2 xiao4 guo3 he2 ying1 wen2 de5 yi2 yang4 hao3']


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = Tacotron(hparams)
      self.model.initialize(inputs, input_lengths)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    # 读取已有模型
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)

  def synthesize(self, text):
    # 将中文转换为注音字符
    text = Pinyin().get_pinyin(text, " ", tone_marks='numbers')
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    # 注音字符到序列
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)}
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()


# 获取输出路径
def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


# 通过最新模型合成语音音频，默认生成文件为./logs-tacotron/eval-100000.wav
def run_eval(ckpt_dir):
  checkpoint = tf.train.get_checkpoint_state(ckpt_dir).model_checkpoint_path
  synth = Synthesizer()
  synth.load(checkpoint)
  base_path = get_output_base_path(checkpoint)
  for i, text in enumerate(sentence):
    path = '%s.wav' % (base_path)
    # print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default='logs-tacotron')
  parser.add_argument('--hparams', default='')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  hparams.parse(args.hparams)
  run_eval(args.checkpoint)


if __name__ == '__main__':
  main()
