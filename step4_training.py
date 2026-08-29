%%writefile /kaggle/working/step4_training.py
"""
STEP 4: Training Loop — now accepts config['encoding_mode'] to select 'direct' or 'poisson'.

NOTE: unlike the decoder ablation (which can be swapped post-hoc on one checkpoint),
input encoding changes what the network sees during training, so a fair comparison
requires training TWO separate models — one per encoding — then comparing them.
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from step1_dataset_lab import LabColorizationDataset, lab_to_rgb_tensor
from step2_snn_generator import SlimGeneratorUNet
from step3_snn_conversion import build_snn_generator


class LabDiscriminator(nn.Module):
    """PatchGAN discriminator. Input: [L, ab] = 3 channels. At 128px input -> 7x7 patch output."""
    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64,  normalize=False),
            *block(64,  128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

    def forward(self, L, ab):
        x = torch.cat([L, ab], dim=1)
        return self.model(x)


def estimate_energy_pJ(spike_count, energy_per_op_pJ=0.9):
    return spike_count * energy_per_op_pJ


def count_ann_macs(H=128, W=128):
    macs = 0
    for (Cin, Cout, H_out) in [(1,64,64), (64,128,32), (128,256,16), (256,256,8), (256,256,4)]:
        macs += Cout * Cin * 4 * 4 * H_out * H_out
    for (Cin, Cout, H_out) in [(256,256,8), (512,256,16), (512,128,32), (256,64,64)]:
        macs += Cout * Cin * 3 * 3 * H_out * H_out
    return macs


def weights_init_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def compute_psnr_ssim(ab_pred, ab_real, L_raw):
    rgb_pred = lab_to_rgb_tensor(L_raw, ab_pred.cpu())
    rgb_real = lab_to_rgb_tensor(L_raw, ab_real.cpu())

    psnr_scores, ssim_scores = [], []
    for i in range(rgb_pred.shape[0]):
        p = rgb_pred[i].numpy().transpose(1, 2, 0)
        r = rgb_real[i].numpy().transpose(1, 2, 0)
        psnr_scores.append(psnr_fn(r, p, data_range=1.0))
        ssim_scores.append(ssim_fn(r, p, channel_axis=2, data_range=1.0))

    return np.mean(psnr_scores), np.mean(ssim_scores)


def save_sample_images(L_norm, ab_pred, ab_real, L_raw, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    rgb_pred = lab_to_rgb_tensor(L_raw[:4], ab_pred[:4].detach().cpu())
    rgb_real = lab_to_rgb_tensor(L_raw[:4], ab_real[:4].cpu())

    for i in range(min(4, L_norm.shape[0])):
        gray_np = ((L_norm[i, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        gen_np  = (rgb_pred[i].numpy().transpose(1,2,0) * 255).clip(0, 255).astype(np.uint8)
        real_np = (rgb_real[i].numpy().transpose(1,2,0) * 255).clip(0, 255).astype(np.uint8)
        gray_rgb = np.stack([gray_np]*3, axis=2)
        combined = np.concatenate([gray_rgb, gen_np, real_np], axis=1)
        Image.fromarray(combined).save(os.path.join(save_dir, f"epoch{epoch:03d}_sample{i}.png"))


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Encoding: {config.get('encoding_mode', 'direct')}")

    train_dataset = LabColorizationDataset(config['train_dir'], image_size=config['image_size'])
    val_dataset   = LabColorizationDataset(config['val_dir'],   image_size=config['image_size'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    snn_gen = build_snn_generator(
        T=config['T_steps'],
        beta=config['lif_beta'],
        pretrained_ann=config.get('pretrained_ann', None),
        encoding=config.get('encoding_mode', 'direct')
    ).to(device)

    discriminator = LabDiscriminator(in_channels=3).to(device)

    snn_gen.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    print(f"SNN Generator params: {sum(p.numel() for p in snn_gen.parameters() if p.requires_grad):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,}")

    criterion_GAN = nn.MSELoss()
    criterion_L1  = nn.L1Loss()

    optimizer_G = optim.Adam(snn_gen.parameters(),     lr=config['lr'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    def lr_lambda(epoch):
        n_epochs_decay = config['num_epochs'] // 2
        if epoch < config['num_epochs'] - n_epochs_decay:
            return 1.0
        return 1.0 - (epoch - (config['num_epochs'] - n_epochs_decay)) / n_epochs_decay

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    os.makedirs(config['metrics_dir'], exist_ok=True)
    csv_path = os.path.join(config['metrics_dir'], 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'G_loss', 'D_loss', 'val_PSNR', 'val_SSIM',
            'mean_firing_rate', 'spike_energy_pJ', 'lambda_spike'
        ])

    best_ssim = 0.0

    for epoch in range(1, config['num_epochs'] + 1):
        snn_gen.train()
        discriminator.train()

        decay_factor = max(0.0, 1.0 - (2.0 * epoch / config['num_epochs']))
        current_lambda_spike = config['lambda_spike'] * decay_factor

        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        epoch_firing_rate = 0.0

        for batch_idx, (L_norm, ab_real, L_raw) in enumerate(train_loader):
            L_norm  = L_norm.to(device)
            ab_real = ab_real.to(device)

            batch_size = L_norm.shape[0]

            # 128px input -> PatchGAN output is 7x7, NOT 15x15 (that was for 256px)
            real_label = torch.ones(batch_size, 1, 7, 7, device=device)
            fake_label = torch.zeros(batch_size, 1, 7, 7, device=device)

            # ---- Train Discriminator ----
            optimizer_D.zero_grad()
            pred_real = discriminator(L_norm, ab_real)
            loss_D_real = criterion_GAN(pred_real, real_label)

            with torch.no_grad():
                ab_fake = snn_gen(L_norm)
            pred_fake = discriminator(L_norm, ab_fake.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake_label)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ---- Train Generator ----
            optimizer_G.zero_grad()
            ab_fake = snn_gen(L_norm)

            pred_fake_for_G = discriminator(L_norm, ab_fake)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, real_label)
            loss_G_L1 = criterion_L1(ab_fake, ab_real)
            loss_spike = snn_gen.last_mean_firing_rate

            loss_G = (loss_G_GAN +
                     config['lambda_L1'] * loss_G_L1 +
                     current_lambda_spike * loss_spike)

            loss_G.backward()
            optimizer_G.step()

            epoch_G_loss    += loss_G.item()
            epoch_D_loss    += loss_D.item()
            epoch_firing_rate += snn_gen.last_mean_firing_rate.item()

        # ---- Validation ----
        snn_gen.eval()
        val_psnr, val_ssim, val_energy = 0.0, 0.0, 0.0
        num_val_batches = 0

        with torch.no_grad():
            for L_norm, ab_real, L_raw in val_loader:
                L_norm = L_norm.to(device)
                ab_pred = snn_gen(L_norm)
                p, s = compute_psnr_ssim(ab_pred, ab_real, L_raw)
                val_psnr += p
                val_ssim += s
                val_energy += estimate_energy_pJ(snn_gen.last_spike_count.item())
                num_val_batches += 1

        val_psnr   /= num_val_batches
        val_ssim   /= num_val_batches
        val_energy /= num_val_batches
        avg_G_loss = epoch_G_loss / len(train_loader)
        avg_D_loss = epoch_D_loss / len(train_loader)
        avg_rate   = epoch_firing_rate / len(train_loader)

        print(f"\nEpoch {epoch}/{config['num_epochs']} [{config.get('encoding_mode','direct')}]")
        print(f"  G Loss : {avg_G_loss:.4f}  |  D Loss: {avg_D_loss:.4f}")
        print(f"  PSNR   : {val_psnr:.4f}  |  SSIM  : {val_ssim:.4f}")
        print(f"  Fire % : {avg_rate:.4f}   |  λ_spike: {current_lambda_spike:.5f}")
        print(f"  Est. Energy (val): {val_energy:.2f} pJ/image")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, avg_G_loss, avg_D_loss, val_psnr, val_ssim,
                avg_rate, val_energy, current_lambda_spike
            ])

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save({
                'epoch': epoch,
                'snn_gen_state': snn_gen.state_dict(),
                'disc_state': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'val_ssim': val_ssim,
                'val_psnr': val_psnr,
                'T_steps': config['T_steps'],
                'lif_beta': config['lif_beta'],
                'encoding_mode': config.get('encoding_mode', 'direct'),
            }, os.path.join(config['checkpoint_dir'], f"best_snn_generator_{config.get('encoding_mode','direct')}.pt"))
            print(f"  *** New best SSIM: {best_ssim:.4f} — checkpoint saved ***")

        if epoch % 5 == 0:
            with torch.no_grad():
                L_sample, ab_real_sample, L_raw_sample = next(iter(val_loader))
                L_sample = L_sample.to(device)
                ab_pred_sample = snn_gen(L_sample)
            save_sample_images(
                L_sample, ab_pred_sample, ab_real_sample, L_raw_sample,
                save_dir=os.path.join(config['samples_dir'], f'epoch_{epoch:03d}'),
                epoch=epoch
            )

        scheduler_G.step()
        scheduler_D.step()

    print("\nTraining complete!")
    print(f"Best SSIM: {best_ssim:.4f}")
    return snn_gen