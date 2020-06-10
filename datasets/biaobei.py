from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio


# 预处理单个<语音，文本>对
def _process_utterance(out_dir, index, wav_path, pinyin):
  # 读取语音
  wav = audio.load_wav(wav_path)
  wav = wav / np.abs(wav).max() * 0.999

  # 消除静音
  wav = audio.trim_silence(wav)

  # 得到语音的线性频谱和梅尔频谱
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # 保存两种频谱
  spectrogram_filename = 'biaobei-spec-%05d.npy' % index
  mel_filename = 'biaobei-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  return (spectrogram_filename, mel_filename, n_frames, pinyin)


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []

  with open(os.path.join(in_dir, 'ProsodyLabeling', '000001-010000.txt'), encoding='utf-8') as f:
    lines = f.readlines()
    index = 1
    sentence_index = ''

    for line in lines:
        if line[0].isdigit():
            sentence_index = line[:6]
        else:
            sentence_pinyin = line.strip()
            wav_path = os.path.join(in_dir, 'Wave', '%s.wav' % sentence_index)
            futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, sentence_pinyin)))
            index = index + 1
  return [future.result() for future in tqdm(futures)]



