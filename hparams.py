import tensorflow as tf


hparams = tf.contrib.training.HParams(
  cleaners='basic_cleaners',

  # 音频处理参数
  num_mels=80,
  num_freq=2049,
  sample_rate=48000,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,
  max_frame_num=1000,
  max_abs_value=4,
  fmin=125,
  fmax=7600,

  # 网络模型参数
  outputs_per_step=5,
  embed_depth=512,
  prenet_depths=[256, 256],
  encoder_depth=256,
  postnet_depth=512,
  attention_depth=128,
  decoder_depth=1024,

  # 训练参数
  batch_size=32,
  adam_beta1=0.9,
  adam_beta2=0.999,
  initial_learning_rate=0.001,
  decay_learning_rate=True,
  use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

  # 声码器（Griffin-Lim）参数
  max_iters=300,
  griffin_lim_iters=60,
  power=1.2)
