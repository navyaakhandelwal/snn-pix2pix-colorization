# SNN-Based Conditional GAN for Image Colorization

An SNN-based conditional GAN for grayscale-to-color image translation using a U-Net generator with LIF spiking neurons, surrogate-gradient training, and membrane-potential decoding.

## Overview

This project investigates the use of Spiking Neural Networks (SNNs) for image-to-image generation.

The model is based on the Pix2Pix conditional GAN framework. The original U-Net generator is converted into a spiking neural network using Leaky Integrate-and-Fire (LIF) neurons. Instead of directly predicting RGB values, the model operates in the CIE Lab color space:

- **L channel:** luminance/structural information
- **a channel:** green ↔ red chrominance
- **b channel:** blue ↔ yellow chrominance

The SNN receives the L channel and predicts the `a,b` chrominance channels. The predicted channels are then combined with the original L channel and converted back to RGB.

## Motivation

Conventional deep neural networks rely heavily on dense multiply-accumulate (MAC) operations. SNNs use sparse spike-based communication and temporal membrane dynamics, making them attractive for low-power and neuromorphic computing applications.

However, image generation presents a challenge for SNNs because pixel and color values are continuous, whereas spikes are binary events.

This project therefore explores:

1. Converting a Pix2Pix U-Net generator into an SNN.
2. Training the SNN using surrogate gradients.
3. Using membrane-potential decoding to obtain continuous-valued color predictions.
4. Studying the trade-off between simulation timesteps and image quality.
5. Comparing membrane-potential, rate, and time-to-first-spike (TTFS) decoding strategies.
6. Estimating inference energy using spike-operation and MAC-operation counts.

## Architecture

```text
                 Grayscale / L channel
                         │
                         ▼
                  U-Net Generator
                         │
                 LIF Spiking Neurons
                         │
                  T simulation steps
                         │
                         ▼
             Membrane-Potential Decoder
                         │
                         ▼
                    a,b channels
                         │
              ┌──────────┴──────────┐
              │                     │
        Original L              Predicted ab
              │                     │
              └──────────┬──────────┘
                         ▼
                   Lab → RGB
                         │
                         ▼
                 Colorized Image
