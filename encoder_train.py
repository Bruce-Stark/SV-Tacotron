from encoder.train import train
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the speaker encoder. You must have run encoder_preprocess.py first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--run_id", type=str, default='SLR68', help="模型名称")
    parser.add_argument("--clean_data_root", type=Path, default=Path('E:/SRP/SV2TTS/encoder_train'),help="预处理后的输出数据路径")
    parser.add_argument("-m", "--models_dir", type=Path, default="E:/SRP/SV2TTS/encoder/saved_models/", help="模型输出路径")
    parser.add_argument("-v", "--vis_every", type=int, default=10, help="每隔多少步更新一次Loss和EER")
    parser.add_argument("-u", "--umap_every", type=int, default=100, help="每隔多少步更新一次UMAP可视化")
    parser.add_argument("-s", "--save_every", type=int, default=500, help="每隔多少步更新一次模型")
    parser.add_argument("-b", "--backup_every", type=int, default=7500, help="每个多少步对模型进行备份")
    parser.add_argument("--visdom_server", type=str, default="http://localhost")

    args = parser.parse_args()
    
    # 处理参数
    args.models_dir.mkdir(exist_ok=True)
    
    # 训练
    train(**vars(args))