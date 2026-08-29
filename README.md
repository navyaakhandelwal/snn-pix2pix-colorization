Absolutely. Replace everything in your `README.md` with the following **complete version**:

````markdown
# SNN-Based Conditional GAN for Image Colorization

A Spiking Neural Network (SNN) implementation of a Pix2Pix conditional GAN for grayscale-to-color image translation, developed as an algorithmic proof-of-concept for potential deployment on neuromorphic hardware.

## Overview

This project converts a standard Pix2Pix conditional GAN into a spiking neural network using Leaky Integrate-and-Fire (LIF) neurons and surrogate-gradient training.

The generator is implemented as a U-Net architecture whose hidden-layer activations are replaced with LIF spiking neurons. The discriminator remains a standard ANN PatchGAN and is used only during training.

The core contribution is a **membrane-potential decoding scheme** that produces continuous-valued output from the SNN. This is particularly useful for image-generation tasks such as colorization, where the desired output consists of continuous pixel/color values rather than binary spike outputs.

The project operates in the CIE Lab color space. The grayscale/luminance `L` channel is provided to the generator, while the network predicts the two chrominance channels `a` and `b`. The predicted chrominance channels are then combined with the original `L` channel and converted back to RGB.

## Motivation

Spiking Neural Networks are attractive for low-power and neuromorphic computing because they represent information using sparse spikes and temporal membrane dynamics.

However, adapting SNNs to generative vision tasks presents a challenge: spikes are binary events, whereas image pixels and color values are continuous.

This project investigates whether a generative image-to-image model can be adapted to an SNN while maintaining useful image quality through membrane-potential decoding.

The project was developed as part of a broader neuromorphic computing research effort involving a custom FPGA/PYNQ-based neuromorphic processor. The current work is an algorithmic/software proof-of-concept; physical hardware deployment is outside the scope of the current implementation.

## Architecture

### Generator

The generator is a 5-level U-Net encoder-decoder operating on `128 × 128` images.

- Input: 1-channel Lab `L` image
- Output: 2-channel `a,b` chrominance prediction
- Encoder-decoder structure with skip connections
- Hidden layers use LIF spiking neurons
- Maximum hidden width: 256 channels
- Approximately 5.2M trainable parameters
- Simulation over `T` timesteps

### Discriminator

The discriminator is an ANN-based PatchGAN.

It receives the `L` channel together with either the real or generated `a,b` channels and produces patch-level real/fake predictions.

The discriminator is used only during training and is not part of the SNN inference pipeline.

### Overall Pipeline

```text
                 Grayscale Image
                       │
                       ▼
                  RGB → Lab
                       │
                       ▼
                    L Channel
                       │
                       ▼
                 U-Net Generator
                       │
                LIF Spiking Layers
                       │
                 T Simulation Steps
                       │
                       ▼
          Membrane-Potential Decoder
                       │
                       ▼
                  Predicted a,b
                       │
             ┌─────────┴─────────┐
             │                   │
        Original L          Predicted a,b
             │                   │
             └─────────┬─────────┘
                       ▼
                    Lab → RGB
                       │
                       ▼
                Colorized Image
````

## Why Lab Color Space?

The model uses the CIE Lab color space instead of directly predicting RGB.

```text
L → luminance
a → green ↔ red chrominance
b → blue ↔ yellow chrominance
```

The `L` channel contains luminance and much of the structural information of the image. Therefore, the network focuses on predicting only the chrominance information.

This also reduces the generator output from three channels to two channels.

## Why U-Net?

U-Net combines an encoder and decoder with skip connections.

The encoder captures increasingly high-level features while reducing spatial resolution. The decoder reconstructs the image resolution.

Skip connections transfer spatial information from the encoder directly to the corresponding decoder layers. This is particularly useful for image-to-image translation because spatial details need to be preserved.

## Why SNN?

Conventional ANNs perform dense numerical computations using continuous activations and multiply-accumulate operations.

SNNs instead communicate using discrete spikes over time. Because spikes can be sparse, SNNs are promising for event-driven and low-power neuromorphic hardware.

The goal of this project is to investigate whether a generative vision model can retain useful image quality after conversion to a spiking architecture.

## LIF Neurons

The hidden layers of the generator use Leaky Integrate-and-Fire neurons.

A simplified membrane update can be represented as:

```text
v[t] = βv[t-1] + input[t]
```

When the membrane potential reaches the firing threshold, the neuron emits a spike and its membrane state is reset.

The main configuration uses:

```text
β = 0.9
Threshold = 1.0
T = 4 timesteps
```

The membrane decay factor controls how much information from previous timesteps is retained.

## Surrogate-Gradient Training

A major difficulty in training SNNs is that the spike-generation function is non-differentiable.

The forward pass still uses the actual binary spike function. During the backward pass, a differentiable surrogate function is used to approximate the gradient.

This allows the SNN to be trained using gradient-based optimization and backpropagation through time.

The implementation uses `snnTorch` and a fast-sigmoid surrogate gradient.

## Membrane-Potential Decoding

The main method explored in this project is membrane-potential decoding.

Instead of decoding the final output only from binary spikes, the membrane potential is accumulated across the simulation timesteps:

```text
v_avg = (v1 + v2 + ... + vT) / T
```

The averaged membrane potential is then passed through a `tanh` activation to obtain the normalized continuous `a,b` prediction.

This allows the SNN to produce continuous-valued color information.

## Decoder Comparison

Three decoding strategies are investigated:

| Decoding Method    | Principle                                                   |
| ------------------ | ----------------------------------------------------------- |
| Rate Coding        | Decode using spike count divided by the number of timesteps |
| TTFS               | Decode using the time of the first spike                    |
| Membrane Potential | Decode using the averaged membrane potential                |

The motivation for membrane-potential decoding is that colorization requires continuous-valued output, while spike-count and first-spike-time representations provide more limited output resolution at low timestep counts.

## Training Objective

The generator is trained using adversarial loss, L1 reconstruction loss, and spike-activity regularization.

Conceptually:

```text
Total Generator Loss
    =
Adversarial Loss
    +
λL1 × L1 Reconstruction Loss
    +
λspike × Spike/Firing Regularization
```

Main training configuration:

```text
Image size      : 128 × 128
Batch size      : 8
Learning rate   : 2 × 10⁻⁴
L1 weight       : 100
Spike penalty   : 0.01
Timesteps (T)  : 4
LIF beta        : 0.9
Epochs          : 50
```

## Results

### Main Result

| Model             | PSNR (dB) |      SSIM |
| ----------------- | --------: | --------: |
| SNN Pix2Pix (T=4) | **21.90** | **0.887** |

### Decoder Comparison

| Decoding Method        | PSNR (dB) |      SSIM |
| ---------------------- | --------: | --------: |
| Rate Coding            |      7.23 |     0.419 |
| TTFS Coding            |      7.07 |     0.423 |
| **Membrane Potential** | **21.90** | **0.887** |

Membrane-potential decoding achieves approximately a 15 dB PSNR improvement over the rate-coding and TTFS decoding configurations evaluated in this project.

### Timestep Ablation

|  T | PSNR (dB) |  SSIM | Estimated Energy (pJ) | Observation                                         |
| -: | --------: | ----: | --------------------: | --------------------------------------------------- |
|  1 |     21.86 | 0.897 |                    ~0 | Collapsed; zero spikes and input-independent output |
|  2 |     21.90 | 0.893 |              ~765,677 | Valid, borderline input sensitivity                 |
|  4 |     21.90 | 0.887 |            ~2,832,473 | Recommended operating point                         |
|  8 |     21.47 | 0.891 |            ~7,400,133 | Valid, diminishing returns                          |

### T=1 Failure Analysis

An important observation from the timestep experiment was that standard image-quality metrics alone can fail to detect a collapsed SNN.

At `T=1`, the model produced:

```text
Firing rate = 0
Mean absolute output difference between
real-image input and random-noise input = 0.0
```

This indicates that the output was independent of the input despite obtaining apparently strong PSNR/SSIM values.

Therefore, the project uses an additional input-sensitivity check to detect this type of failure.

## Energy Estimation

The project estimates inference energy using operation counts rather than physical hardware measurements.

For the SNN:

```text
Estimated SNN Energy
≈ Number of spike/synaptic operations × Energy per SOP
```

For the ANN:

```text
Estimated ANN Energy
≈ Number of MAC operations × Energy per MAC
```

The implementation uses assumed per-operation energy values to provide a theoretical comparison.

**Important:** the reported energy values are estimates and are not measurements from a physical neuromorphic processor. Actual hardware energy would depend on the target architecture, memory access, communication, implementation details, clocking, and other system-level factors.

## Dataset

The project uses a landscape-image dataset consisting of:

```text
Training images   : 3,500
Validation images : 400
Test images       : 419
Resolution        : 128 × 128
```

The dataset is used only for training and evaluation and is not included in this repository.

## Project Structure

```text
snn-pix2pix-colorization/
│
├── src/
│   ├── step1_dataset_lab.py
│   ├── step2_snn_generator.py
│   ├── step3_snn_conversion.py
│   ├── step4_training.py
│   └── step5_evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Source Files

#### `step1_dataset_lab.py`

Handles:

* Image loading
* Resizing
* RGB → Lab conversion
* L/ab channel separation
* Normalization
* Lab → RGB reconstruction

#### `step2_snn_generator.py`

Defines the U-Net generator architecture used as the ANN backbone before SNN conversion.

#### `step3_snn_conversion.py`

Implements:

* LIF neuron dynamics
* Surrogate-gradient support
* SNN generator wrapper
* Multi-timestep simulation
* Membrane-potential decoding
* Rate decoding
* TTFS decoding

#### `step4_training.py`

Contains:

* PatchGAN discriminator
* Generator training
* Discriminator training
* Adversarial loss
* L1 reconstruction loss
* Spike regularization
* PSNR/SSIM evaluation
* Energy estimation

#### `step5_evaluation.py`

Contains:

* Main model evaluation
* Timestep ablation
* Decoder ablation
* PSNR/SSIM comparison
* FID evaluation
* Energy estimation

## Installation

Clone the repository:

```bash
git clone https://github.com/navyaakhandelwal/snn-pix2pix-colorization.git
cd snn-pix2pix-colorization
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Configure the dataset and training paths in the training configuration.

The training pipeline can then be started with:

```bash
python src/step4_training.py
```

Evaluation can be performed using:

```bash
python src/step5_evaluation.py
```

The dataset itself is not included in this repository.

## Sample Output

Example visualization:

```text
Input Grayscale → SNN Colorized Output → Ground Truth
```

Sample result images can be added to the repository under a `results/` directory.

## Limitations

* The current implementation is software-based SNN simulation.
* The target neuromorphic processor is not directly integrated with this implementation.
* Energy values are theoretical operation-based estimates rather than physical measurements.
* Image quality is below the corresponding ANN baseline, representing a quality-efficiency tradeoff.
* The model is trained exclusively on landscape imagery and therefore may not generalize to other scene types without retraining.
* The current experiments use relatively low-resolution `128 × 128` images.
* Physical neuromorphic deployment would require hardware-specific mapping and optimization.

## Future Work

* Deploy the SNN generator on the target neuromorphic processor.
* Measure actual hardware power and latency.
* Investigate hardware-aware training and optimization.
* Explore quantization and reduced-precision inference.
* Evaluate higher image resolutions.
* Evaluate the model on additional image domains.
* Explore alternative SNN architectures and decoding mechanisms.
* Investigate more efficient spike encoding and temporal representations.

## Tech Stack

* Python
* PyTorch
* snnTorch
* NumPy
* Pillow
* scikit-image
* TorchMetrics

### Concepts

* Convolutional Neural Networks
* U-Net
* Conditional GANs
* Pix2Pix
* PatchGAN
* Spiking Neural Networks
* LIF Neurons
* Surrogate Gradients
* Neuromorphic Computing

## Acknowledgements

This project was developed as part of a broader neuromorphic computing research effort involving a custom FPGA/PYNQ-based neuromorphic processor developed alongside the project.
