import numpy as np
from tqdm import tqdm
from encoder import audio_preprocess
from pathlib import Path
from multiprocess.pool import Pool


# 采样率
sampling_rate = 16000
# 话语中的谱图帧数
partials_n_frames = 160     # 1600 ms
inference_n_frames = 80     # 800 ms


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension, skip_existing, logger):
    print("%s: Preprocessing data for %d speakers." % (dataset_name, len(speaker_dirs)))

    # 预处理一个说话人的所有语音
    def preprocess_speaker(speaker_dir: Path):
        speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")
        sources_file = sources_fpath.open("a" if skip_existing else "w")
        for in_fpath in speaker_dir.glob("**/*.%s" % extension):
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")

            # 预处理单条语音
            wav = audio.preprocess_wav(in_fpath)

            # 丢弃过短语音
            frames = audio.wav_to_mel_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue

            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))

        sources_file.close()

    # 处理每个说话人的所有语音
    with Pool(32) as pool:
        list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs), unit="speakers"))
    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)


# 预处理数据集SLR38, 数据集下载地址: http://www.openslr.org/38/
def preprocess_SLR38(datasets_root: Path, out_dir: Path, skip_existing=False):
    dataset_name = "SLR38/wav"
    dataset_root = datasets_root.joinpath(dataset_name)
    all_sub_dirs = list(dataset_root.glob('*'))
    speaker_dirs = []
    for _dir in all_sub_dirs:
        if _dir.is_file(): continue
        speaker_dirs.append(_dir)
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, 'wav', skip_existing)


# 预处理数据集SLR68, 数据集下载地址: http://www.openslr.org/68/
def preprocess_SLR68(datasets_root: Path, out_dir: Path, skip_existing=False):
    dataset_name = 'SLR68/train'
    dataset_root = datasets_root.joinpath(dataset_name)
    all_sub_dirs = list(dataset_root.glob('*'))
    speaker_dirs = []
    for _dir in all_sub_dirs:
        if _dir.is_file(): continue
        speaker_dirs.append(_dir)
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, 'wav', skip_existing)
