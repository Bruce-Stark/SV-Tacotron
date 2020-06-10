from encoder.preprocess import preprocess_SLR68, preprocess_SLR38
from pathlib import Path
import argparse


if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms and writes them to the disk.",
        formatter_class=MyFormatter
    )
    parser.add_argument("--datasets_root", type=Path, default=Path('E:/SRP/SLR68/dataset'),
                        help="SLR38和SLR68数据库的路径")
    parser.add_argument("-d", "--datasets", type=str, default='SLR68',
                        help="数据库名称（SLR68或者SLR38）")
    parser.add_argument("-o", "--out_dir", type=Path, default="E:/SRP/SLR68/dataset/SV2TTS/encoder_train",
                        help="梅尔频谱输出路径")
    args = parser.parse_args()

    # 处理参数
    args.datasets = args.datasets.split(",")
    args.out_dir = args.datasets_root.joinpath("SV2TTS", args.out_dir)
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # 数据库预处理
    preprocess_func = {
        "SLR68": preprocess_SLR68,
        "SLR38": preprocess_SLR38,
    }
    args = vars(args)
    for dataset in args.pop("datasets"):
        print("Preprocessing %s" % dataset)
        preprocess_func[dataset](**args)
