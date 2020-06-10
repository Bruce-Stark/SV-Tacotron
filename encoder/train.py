from encoder.visualizations import Visualizations
from datasets.speaker_verification_dataset import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
import torch

# 模型参数
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

# 训练参数
learning_rate_init = 1e-4
speakers_per_batch = 64
utterances_per_speaker = 10


def sync(device: torch.device):
    return 
    # cuda同步
    # if device.type == "cuda":
    #     torch.cuda.synchronize(device)


def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):

    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=8,
    )

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")
    
    # 创建模型和优化器
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1
    
    # 为模型配置文件路径
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    model.train()
    
    # 初始化可视化环境（visdom）
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    
    # 开始训练
    profiler = Profiler(summarize_every=10, disabled=False)
    for step, speaker_batch in enumerate(loader, init_step):
        profiler.tick("Blocking, waiting for batch (threaded)")
        
        # 正向传播
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")

        # 反向传播
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Parameter update")
        vis.update(loss.item(), eer, step)
        
        # 进行一次UMAP投影可视化并保存图片
        if umap_every != 0 and step % umap_every == 0:
            # print("Drawing and saving projections (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
            vis.save()

        # 更新模型
        if save_every != 0 and step % save_every == 0:
            # print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
        # 进行一次备份
        if backup_every != 0 and step % backup_every == 0:
            # print("Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
            
        profiler.tick("Extras (visualizations, saving)")