"""PyTorch implementation of an AWGN wireless channel.
    Source: https://github.com/brysef/rfml/blob/master/rfml/ptradio/awgn.py
"""

# External Includes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_energy(x):
    real, imag = x.chunk(x.size()[-2], dim=-2)
    real = real ** 2
    imag = imag ** 2
    # Energy = np.abs(s)^2 => np.sqrt(r^2 + i^2)^2
    # Therefore:
    energy = real + imag
    energy = torch.squeeze(torch.mean(energy, dim=-1))
    return energy

class AWGN(nn.Module):
    """Additive White Gaussian Noise (AWGN) Channel model implemented in PyTorch.

    The noise power is chosen uniformly at random between snr_low and snr_high, 
    which can be updated by calling *set_snr*.
    Each forward pass will have a different noise realization.  
    This layer has no effect on sizes and can be made to be a pass through 
    by setting SNR to None.

    Args:
        snr_min (float, optional): minimum Signal-to-Noise ratio.  This can be overriden during
                               operation by calling set_snr.  Defaults to None.
        snr_max (float, optional): maximum Signal-to-Noise ratio.

    This module makes no assumptions about the shape of the input and returns an
    identically shaped output.
    """

    def __init__(self, snr_min: int = None, snr_max: int = None):
        super(AWGN, self).__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.snr_min is None or self.snr_max is None:
            return x
        else:
            real, imag = x.chunk(x.size()[2], dim=-2)

            real = real ** 2
            imag = imag ** 2
            sig_power = torch.squeeze(torch.mean(real + imag, dim=-1))
            sig_power = 10.* torch.log10(sig_power)
            # The power has to be divided by two here because the PyTorch noise
            # generation is not "complex valued" and therefore we have to split the
            # power evenly between I and Q
            snr = torch.randint(int(self.snr_min), int(self.snr_max), size=(x.size(0),), device="cuda")
            noise_power = sig_power - snr
            noise_power = torch.pow(torch.pow(10.0, noise_power / 10.0) / 2.0, 0.5)
            noise_power = torch.tile(noise_power, (x.size(-1),2, 1))
            noise_power = torch.transpose(noise_power, 0, -1)
            noise_power = noise_power[:, np.newaxis, :, :]

            noise = noise_power * x.data.new(x.size()).normal_(0.0,1.0)
            return x + noise

    def set_snr(self, snr_min: float, snr_max: float):
        """Set the signal to noise ratio in dB"""
        self.snr_min = snr_min
        self.snr_max = snr_max