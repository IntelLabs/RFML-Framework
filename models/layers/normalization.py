'''
.. module:: normalization
    :platform: Unix (Ubuntu)
    :synopsis: Custom PyTorch Normalization Layers

.. moduleauthor:: Bryse Flowers <brysef@vt.edu>
'''
__author__ = "Bryse Flowers"

# PyTorch Includes
import torch
import torch.nn as nn


class EnergyNormalization(nn.Module):
    '''Perform average energy per sample normalization.

    Energy Normalization is performed as follows:

        energy = np.mean(np.abs(signal)**2) # Scalar
        signal = signal / np.sqrt(energy) # Normalized signal

    This module assumes that the signal is structured as:
        B x C x I/Q x T

    Where the energy normalization is performed along the T axis using the
    energy measured in the complex I/Q dimension.
    '''

    def __init__(self):
        super(EnergyNormalization, self).__init__()

    def forward(self, x):
        real, imag = x.chunk(x.size()[2], dim=2)

        real = real ** 2
        imag = imag ** 2
        # Energy = np.abs(s)^2 => np.sqrt(r^2 + i^2)^2
        # Therefore:
        energy = real + imag

        energy = energy.mean(dim=3)
        energy = energy.view([energy.size()[0], energy.size()[1],
                energy.size()[2], 1])

        return x / torch.sqrt(energy)


class EnergyNormalizationMax(nn.Module):
    '''Perform max energy per sample normalization.

    Energy Normalization is performed as follows:

        energy = np.abs(energy) # Scalar
        signal = signal / np.max(energy) # Normalized signal

    This module assumes that the signal is structured as:
        B x C x I/Q x T

    Where the energy normalization is performed along the T axis using the
    energy measured in the complex I/Q dimension.
    '''

    def __init__(self):
        super(EnergyNormalizationMax, self).__init__()

    def forward(self, x):
        real, imag = x.chunk(x.size()[2], dim=-2)

        real = real ** 2
        imag = imag ** 2
        # Energy = np.abs(s)^2 => np.sqrt(r^2 + i^2)^2
        # Therefore:
        energy = real + imag

        energy = torch.abs(energy)
        return x / torch.max(energy,dim=3)
