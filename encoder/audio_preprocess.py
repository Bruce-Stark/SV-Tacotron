from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa
from utils import logmmse
from scipy.io import wavfile

int16_max = (2 ** 15) - 1

# 梅尔过滤器组
mel_window_length = 25  # 单位：毫秒
mel_window_step = 10    # 单位：毫秒
mel_n_channels = 40
# 采样率
sampling_rate = 16000
# 音频音量标准化
audio_norm_target_dBFS = -30
# 话语中的谱图帧数
partials_n_frames = 160     # 1600 ms
inference_n_frames = 80     # 800 ms
# 静音检测算法（VAD）
vad_window_length = 30  # 单位：毫秒
vad_moving_average_width = 8
vad_max_silence_length = 6


# 保存音频
def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


# 去除静音
def trim_silence(wav, top_db=60):
    return librosa.effects.trim(wav, top_db=top_db, frame_length=512, hop_length=128)[0]


# 规范音量
def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


# 语音信号转为梅尔频谱
def wav_to_mel_spectrogram(wav):
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


# 语音预处理（去噪，去除静音）
def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None):

    # 读取音频
    wav, source_sr = librosa.load(str(fpath_or_wav), sr=sampling_rate)
    wav_abs_max = np.max(np.abs(wav))
    wav_abs_max = wav_abs_max if wav_abs_max > 0.0 else 1e-8
    wav = wav / wav_abs_max * 0.9

    # 去噪
    if len(wav) > sampling_rate*(0.3+0.1):
        noise_wav = np.concatenate([wav[:int(sampling_rate*0.15)],
                                    wav[-int(sampling_rate*0.15):]])
        profile = logmmse.profile_noise(noise_wav, sampling_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # 去除静音
    wav = librosa.effects.trim(wav, top_db=30, frame_length=512, hop_length=128)[0]
    return wav








