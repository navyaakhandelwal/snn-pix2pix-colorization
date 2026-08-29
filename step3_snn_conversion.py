%%writefile /kaggle/working/step3_snn_conversion.py
"""
STEP 3: SNN Conversion — LIF Neurons + Surrogate Gradient + INPUT ENCODING ABLATION
======================================================================================

INPUT ENCODING MODES:
- 'direct'  (baseline): same analog L image fed at every timestep (constant current).
- 'poisson' (new): at each timestep t, generate a fresh Bernoulli spike per pixel,
  with firing probability proportional to pixel intensity. Brighter pixel -> more
  likely to spike at any given t. This is the standard "rate coding" input scheme
  used across most SNN literature — we're testing whether it helps or hurts a
  dense regression task (colorization) compared to feeding the raw analog signal.

OUTPUT DECODING (unchanged): membrane potential decoder, averaged over T timesteps,
then Tanh. This ablation only touches the INPUT side.
"""

import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate
    SNNTORCH_AVAILABLE = True
except ImportError:
    print("SNNTorch not installed. Run: pip install snntorch")
    SNNTORCH_AVAILABLE = False

from step2_snn_generator import SlimGeneratorUNet


BETA = 0.9
V_THRESHOLD = 1.0
T_STEPS = 4
ENCODING_MODE = 'direct'   # 'direct' or 'poisson' — set per-run in build_snn_generator()

SURROGATE_GRAD = surrogate.fast_sigmoid(slope=25) if SNNTORCH_AVAILABLE else None


class SNNGeneratorWrapper(nn.Module):
    """
    Wraps SlimGeneratorUNet, replaces hidden activations with LIF neurons,
    and supports pluggable INPUT encoding schemes.
    """
    def __init__(self, ann_generator, T=T_STEPS, beta=BETA, v_threshold=V_THRESHOLD,
                 encoding=ENCODING_MODE):
        super().__init__()

        self.T = T
        self.beta = beta
        self.v_threshold = v_threshold
        self.encoding = encoding
        assert encoding in ('direct', 'poisson'), f"Unknown encoding: {encoding}"

        self.generator = ann_generator
        self.lif_neurons = nn.ModuleDict()
        self._register_lif_neurons()

    def _register_lif_neurons(self):
        if not SNNTORCH_AVAILABLE:
            raise RuntimeError("snntorch required. Run: pip install snntorch")

        for i in range(1, 6):
            self.lif_neurons[f'down{i}'] = snn.Leaky(
                beta=self.beta, threshold=self.v_threshold,
                spike_grad=SURROGATE_GRAD, reset_mechanism='subtract', init_hidden=False
            )
        for i in range(1, 5):
            self.lif_neurons[f'up{i}'] = snn.Leaky(
                beta=self.beta, threshold=self.v_threshold,
                spike_grad=SURROGATE_GRAD, reset_mechanism='subtract', init_hidden=False
            )

    def _init_mem(self, batch_size, device):
        mems = {}
        for name, lif in self.lif_neurons.items():
            mems[name] = lif.init_leaky()
        return mems

    def _encode_input(self, x):
        """
        x: [B, 1, H, W], normalized L channel, range [-1, 1]

        'direct'  -> return x unchanged (same value fed every timestep)
        'poisson' -> convert to a firing probability in [0,1], sample a fresh
                     Bernoulli spike this timestep, rescale back to [-1,1] so the
                     downstream conv/LIF layers see input on the same numeric scale.
        """
        if self.encoding == 'direct':
            return x
        elif self.encoding == 'poisson':
            prob = ((x + 1.0) / 2.0).clamp(0.0, 1.0)   # [0,1] firing probability
            spikes = torch.bernoulli(prob)              # fresh sample every call
            return spikes * 2.0 - 1.0                   # back to [-1,1] scale
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def forward_one_timestep(self, x, mems):
        x_t = self._encode_input(x)   # <-- encoding applied fresh each timestep
        g = self.generator

        pre_d1 = g.down1.model[:-1](x_t)
        spk_d1, mems['down1'] = self.lif_neurons['down1'](pre_d1, mems['down1'])

        pre_d2 = g.down2.model[:-1](spk_d1)
        spk_d2, mems['down2'] = self.lif_neurons['down2'](pre_d2, mems['down2'])

        pre_d3 = g.down3.model[:-1](spk_d2)
        spk_d3, mems['down3'] = self.lif_neurons['down3'](pre_d3, mems['down3'])

        pre_d4 = g.down4.model[:-1](spk_d3)
        spk_d4, mems['down4'] = self.lif_neurons['down4'](pre_d4, mems['down4'])

        pre_d5 = g.down5.model[:-1](spk_d4)
        spk_d5, mems['down5'] = self.lif_neurons['down5'](pre_d5, mems['down5'])

        pre_u1 = g.up1.model[:-1](spk_d5)
        spk_u1, mems['up1'] = self.lif_neurons['up1'](pre_u1, mems['up1'])

        pre_u2 = g.up2.model[:-1](torch.cat([spk_u1, spk_d4], dim=1))
        spk_u2, mems['up2'] = self.lif_neurons['up2'](pre_u2, mems['up2'])

        pre_u3 = g.up3.model[:-1](torch.cat([spk_u2, spk_d3], dim=1))
        spk_u3, mems['up3'] = self.lif_neurons['up3'](pre_u3, mems['up3'])

        pre_u4 = g.up4.model[:-1](torch.cat([spk_u3, spk_d2], dim=1))
        spk_u4, mems['up4'] = self.lif_neurons['up4'](pre_u4, mems['up4'])

        u5_in = torch.cat([spk_u4, spk_d1], dim=1)
        u5 = g.final_up(u5_in)
        out_pre_tanh = g.final_conv(u5)  # raw membrane potential, NOT spiked

        total_spikes = (spk_d1.sum() + spk_d2.sum() + spk_d3.sum() +
                        spk_d4.sum() + spk_d5.sum() +
                        spk_u1.sum() + spk_u2.sum() + spk_u3.sum() + spk_u4.sum())

        return out_pre_tanh, mems, total_spikes

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        mems = self._init_mem(batch_size, device)

        accumulated_output = torch.zeros(batch_size, 2, x.shape[2], x.shape[3], device=device)
        total_spike_count = 0.0

        for t in range(self.T):
            out_t, mems, spk_count = self.forward_one_timestep(x, mems)
            accumulated_output = accumulated_output + out_t
            total_spike_count = total_spike_count + spk_count

        avg_output = accumulated_output / self.T
        ab_pred = torch.tanh(avg_output)

        self.last_spike_count = total_spike_count
        self.last_num_neurons = self._count_spiking_neurons(x.shape[2], x.shape[3])
        self.last_mean_firing_rate = total_spike_count / (self.last_num_neurons * self.T * batch_size + 1e-8)

        return ab_pred

    def _count_spiking_neurons(self, H, W):
        total = 0
        for level, (ch, spatial) in enumerate([(64,64),(128,32),(256,16),(256,8),(256,4)]):
            total += ch * spatial * spatial
        for level, (ch, spatial) in enumerate([(256,8),(256,16),(128,32),(64,64)]):
            total += ch * spatial * spatial
        return total


class RateCodingDecoder(nn.Module):
    """Alternative OUTPUT decoder (rate coding) — kept from your decoder ablation, unchanged."""
    def forward(self, spike_accumulation, T):
        rate = spike_accumulation / T
        return 2.0 * rate - 1.0


def build_snn_generator(T=4, beta=0.9, pretrained_ann=None, encoding='direct'):
    """
    Build the SNN generator.

    Args:
        T: number of timesteps
        beta: membrane decay rate
        pretrained_ann: optional path to pretrained ANN checkpoint
        encoding: 'direct' (baseline, constant current) or 'poisson' (rate-coded spikes)
    """
    ann_gen = SlimGeneratorUNet(in_ch=1, out_ch=2)

    if pretrained_ann is not None:
        print(f"Loading pretrained ANN weights from {pretrained_ann}")
        ckpt = torch.load(pretrained_ann, map_location='cpu', weights_only=False)
        model_dict = ann_gen.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        loaded = len(pretrained_dict)
        model_dict.update(pretrained_dict)
        ann_gen.load_state_dict(model_dict, strict=False)
        print(f"Loaded {loaded}/{len(model_dict)} layers from pretrained ANN.")

    snn_gen = SNNGeneratorWrapper(ann_gen, T=T, beta=beta, encoding=encoding)
    return snn_gen


if __name__ == "__main__":
    if not SNNTORCH_AVAILABLE:
        print("Install snntorch first: pip install snntorch")
        exit()

    for enc in ['direct', 'poisson']:
        print(f"\n--- Testing encoding = {enc} ---")
        snn_gen = build_snn_generator(T=4, encoding=enc)
        snn_gen = snn_gen.cuda() if torch.cuda.is_available() else snn_gen
        device = next(snn_gen.parameters()).device
        dummy_L = torch.randn(2, 1, 128, 128).to(device)

        with torch.no_grad():
            ab_pred = snn_gen(dummy_L)

        print("Output shape:", ab_pred.shape)
        print("Output range:", ab_pred.min().item(), "to", ab_pred.max().item())
        print(f"Mean firing rate: {snn_gen.last_mean_firing_rate:.4f}")

    print("\nStep 3 complete: both encoding modes working.")