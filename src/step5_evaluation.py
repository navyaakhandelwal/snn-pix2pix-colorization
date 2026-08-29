%%writefile /kaggle/working/step5_evaluation.py
"""
STEP 5: Evaluation + Ablation Studies

TABLE 4 (NEW) — Ablation C: Input encoding comparison
  Encoding | PSNR | SSIM | Mean firing rate | Energy(pJ)
  Direct (baseline) | ... | ... | ... | ...
  Poisson rate       | ... | ... | ... | ...

Requires TWO separately trained checkpoints (one per encoding), since encoding
is baked into what the network learns, unlike the decoder ablation.
"""

import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from step1_dataset_lab import LabColorizationDataset, lab_to_rgb_tensor
from step2_snn_generator import SlimGeneratorUNet
from step3_snn_conversion import build_snn_generator, SNNGeneratorWrapper
from step4_training import LabDiscriminator, compute_psnr_ssim, estimate_energy_pJ

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


class TTFSDecoder:
    @staticmethod
    def decode_from_membrane_trajectory(mem_trajectory, v_threshold=1.0, T=8):
        B, C, H, W = mem_trajectory[0].shape
        ttfs_map = torch.full((B, C, H, W), fill_value=float(T), device=mem_trajectory[0].device)
        for t, mem in enumerate(mem_trajectory):
            fired_now = (mem >= v_threshold)
            never_fired = (ttfs_map == float(T))
            update_mask = fired_now & never_fired
            ttfs_map[update_mask] = float(t)
        ttfs_value = 1.0 - (ttfs_map / T)
        return 2.0 * ttfs_value - 1.0


class AblationSNNGenerator(SNNGeneratorWrapper):
    def forward_with_ablation(self, x):
        batch_size = x.shape[0]
        device = x.device
        mems = self._init_mem(batch_size, device)

        accumulated_output = torch.zeros(batch_size, 2, x.shape[2], x.shape[3], device=device)
        accumulated_spikes = torch.zeros(batch_size, 2, x.shape[2], x.shape[3], device=device)
        mem_trajectory = []

        for t in range(self.T):
            out_t, mems, spk_count = self.forward_one_timestep(x, mems)
            accumulated_output += out_t
            spk_t = (out_t >= 0.5).float()
            accumulated_spikes += spk_t
            mem_trajectory.append(out_t.clone())

        avg_mem = accumulated_output / self.T
        ab_membpot = torch.tanh(avg_mem)

        rate = accumulated_spikes / self.T
        ab_rate = 2.0 * rate - 1.0

        ab_ttfs = TTFSDecoder.decode_from_membrane_trajectory(mem_trajectory, T=self.T)

        return ab_membpot, ab_rate, ab_ttfs


def compute_fid(model, test_loader, device, max_batches=50):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid.reset()
    model.eval()

    with torch.no_grad():
        for i, (L_norm, ab_real, L_raw) in enumerate(test_loader):
            if i >= max_batches:
                break
            L_norm = L_norm.to(device)
            ab_pred = model(L_norm)
            rgb_pred = lab_to_rgb_tensor(L_raw, ab_pred.cpu()).to(device)
            rgb_real = lab_to_rgb_tensor(L_raw, ab_real).to(device)
            fid.update((rgb_real * 255).to(torch.uint8), real=True)
            fid.update((rgb_pred.clamp(0,1) * 255).to(torch.uint8), real=False)

    return fid.compute().item()


def evaluate_main(model, test_loader, device, model_name="SNN"):
    model.eval()
    all_psnr, all_ssim, all_energy = [], [], []

    with torch.no_grad():
        for L_norm, ab_real, L_raw in test_loader:
            L_norm = L_norm.to(device)
            ab_pred = model(L_norm)
            p, s = compute_psnr_ssim(ab_pred, ab_real, L_raw)
            all_psnr.append(p)
            all_ssim.append(s)
            all_energy.append(estimate_energy_pJ(model.last_spike_count.item()))

    results = {
        'model': model_name,
        'PSNR':   np.mean(all_psnr),
        'SSIM':   np.mean(all_ssim),
        'Energy_pJ': np.mean(all_energy)
    }
    print(f"[{model_name}] PSNR={results['PSNR']:.4f} | SSIM={results['SSIM']:.4f} | "
          f"Energy={results['Energy_pJ']:.2f} pJ")
    return results


def ablation_timesteps(checkpoint_path, test_loader, device, T_values=[1, 2, 4, 8], encoding='direct'):
    print("\n=== Ablation A: Timestep T ===")
    results = []
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    for T in T_values:
        print(f"\n--- T = {T} ---")
        model = build_snn_generator(T=T, encoding=encoding).to(device)
        model.load_state_dict(ckpt['snn_gen_state'], strict=False)
        model.T = T
        res = evaluate_main(model, test_loader, device, model_name=f"SNN_T{T}")
        res['T'] = T
        results.append(res)

    print("\n--- Table 2: Ablation A Results ---")
    print(f"{'T':>4} | {'PSNR':>7} | {'SSIM':>7} | {'Energy(pJ)':>12}")
    print("-" * 40)
    for r in results:
        print(f"{r['T']:>4} | {r['PSNR']:>7.4f} | {r['SSIM']:>7.4f} | {r['Energy_pJ']:>12.2f}")
    return results


def ablation_decoder(checkpoint_path, test_loader, device):
    print("\n=== Ablation B: Decoder Comparison ===")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ann_gen = SlimGeneratorUNet(in_ch=1, out_ch=2)
    model = AblationSNNGenerator(ann_gen, T=ckpt.get('T_steps', 4),
                                  encoding=ckpt.get('encoding_mode', 'direct'))
    model.load_state_dict(ckpt['snn_gen_state'], strict=False)
    model = model.to(device)
    model.eval()

    results = {
        'membpot': {'psnr': [], 'ssim': []},
        'rate':    {'psnr': [], 'ssim': []},
        'ttfs':    {'psnr': [], 'ssim': []},
    }

    with torch.no_grad():
        for L_norm, ab_real, L_raw in test_loader:
            L_norm = L_norm.to(device)
            ab_membpot, ab_rate, ab_ttfs = model.forward_with_ablation(L_norm)
            for ab_pred, key in [(ab_membpot, 'membpot'), (ab_rate, 'rate'), (ab_ttfs, 'ttfs')]:
                p, s = compute_psnr_ssim(ab_pred, ab_real, L_raw)
                results[key]['psnr'].append(p)
                results[key]['ssim'].append(s)

    print("\n--- Table 3: Ablation B Results ---")
    print(f"{'Decoder':>26} | {'PSNR':>7} | {'SSIM':>7}")
    print("-" * 46)
    decoder_names = {'rate': 'Rate coding', 'ttfs': 'TTFS coding', 'membpot': 'Membrane potential (ours)'}
    for key in ['rate', 'ttfs', 'membpot']:
        p = np.mean(results[key]['psnr'])
        s = np.mean(results[key]['ssim'])
        print(f"{decoder_names[key]:>26} | {p:>7.4f} | {s:>7.4f}")
    return results


def ablation_encoding(checkpoint_paths, test_loader, device):
    """
    NEW — Table 4: Ablation C, Input encoding comparison.
    checkpoint_paths: dict like {'direct': 'path/to/direct.pt', 'poisson': 'path/to/poisson.pt'}
    Each checkpoint must have been TRAINED with that encoding (not swapped post-hoc).
    """
    print("\n=== Ablation C: Input Encoding Comparison ===")
    results = []

    for enc_name, ckpt_path in checkpoint_paths.items():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_snn_generator(T=ckpt.get('T_steps', 4), encoding=enc_name).to(device)
        model.load_state_dict(ckpt['snn_gen_state'])
        model.eval()

        all_psnr, all_ssim, all_energy, all_rate = [], [], [], []
        with torch.no_grad():
            for L_norm, ab_real, L_raw in test_loader:
                L_norm = L_norm.to(device)
                ab_pred = model(L_norm)
                p, s = compute_psnr_ssim(ab_pred, ab_real, L_raw)
                all_psnr.append(p)
                all_ssim.append(s)
                all_energy.append(estimate_energy_pJ(model.last_spike_count.item()))
                all_rate.append(model.last_mean_firing_rate.item())

        res = {
            'encoding': enc_name,
            'PSNR': np.mean(all_psnr),
            'SSIM': np.mean(all_ssim),
            'firing_rate': np.mean(all_rate),
            'Energy_pJ': np.mean(all_energy),
        }
        results.append(res)
        print(f"[{enc_name}] PSNR={res['PSNR']:.4f} | SSIM={res['SSIM']:.4f} | "
              f"Firing rate={res['firing_rate']:.4f} | Energy={res['Energy_pJ']:.2f} pJ")

    print("\n--- Table 4: Ablation C Results (Input Encoding) ---")
    print(f"{'Encoding':>12} | {'PSNR':>7} | {'SSIM':>7} | {'Fire rate':>10} | {'Energy(pJ)':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['encoding']:>12} | {r['PSNR']:>7.4f} | {r['SSIM']:>7.4f} | "
              f"{r['firing_rate']:>10.4f} | {r['Energy_pJ']:>12.2f}")

    return results