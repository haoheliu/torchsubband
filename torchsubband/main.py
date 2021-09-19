#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/19/21 4:07 PM   Haohe Liu      1.0         None
'''

from torchlibrosa.stft import STFT, ISTFT, magphase
import torch
import torch.nn as nn
import numpy as np
from torchsubband.pqmf import PQMF
import torch.nn.functional as F
from math import ceil

class SubbandDSP(nn.Module):
    def __init__(self,
             subband = 2,
             window_size=2048,
             hop_size=441,
             center=True,
             pad_mode='reflect',
             window='hann',
             freeze_parameters = True,
             ):

        super(SubbandDSP, self).__init__()
        self.subband = subband
        self.stft = STFT(n_fft=window_size // self.subband, hop_length=hop_size // self.subband,
                         win_length=window_size // self.subband, window=window, center=center,
                         pad_mode=pad_mode, freeze_parameters=freeze_parameters)

        self.istft = ISTFT(n_fft=window_size // self.subband, hop_length=hop_size // self.subband,
                           win_length=window_size // self.subband, window=window, center=center,
                           pad_mode=pad_mode, freeze_parameters=freeze_parameters)

        if(subband > 1):
            self.qmf = PQMF(subband, 64)

    def _complex_spectrogram(self, input):
        # [batchsize, samples]
        # return [batchsize, 2, t-steps, f-bins]
        real, imag = self.stft(input)
        return torch.cat([real, imag], dim=1)

    def _reverse_complex_spectrogram(self, input, length=None):
        # [batchsize, 2[real,imag], t-steps, f-bins]
        wav = self.istft(input[:, 0:1, ...], input[:, 1:2, ...], length=length)
        return wav

    def _spectrogram(self, input):
        (real, imag) = self.stft(input.float())
        return torch.clamp(real ** 2 + imag ** 2, 1e-8, np.inf) ** 0.5

    def _spectrogram_phase(self, input):
        (real, imag) = self.stft(input.float())
        mag = torch.clamp(real ** 2 + imag ** 2, 1e-8, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = self._spectrogram_phase(input[:, channel, :])
            sp_list.append(mag)
            cos_list.append(cos)
            sin_list.append(sin)

        sps = torch.cat(sp_list, dim=1)
        coss = torch.cat(cos_list, dim=1)
        sins = torch.cat(sin_list, dim=1)
        return sps, coss, sins

    def spectrogram_phase_to_wav(self, sps, coss, sins, length):
        channels_num = sps.size()[1]
        res = []
        for i in range(channels_num):
            res.append(self.istft(sps[:,i:i+1,...] * coss[:,i:i+1,...], sps[:,i:i+1,...] * sins[:,i:i+1,...], length))
            res[-1] = res[-1].unsqueeze(1)
        return torch.cat(res,dim=1)

    def wav_to_spectrogram(self, input):
        """Waveform to spectrogram.

        Args:
          input: (batch_size,channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            sp_list.append(self._spectrogram(input[:, channel, :]))
        output = torch.cat(sp_list, dim=1)
        return output

    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.
        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[1]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, channel, :])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(self.istft(spectrogram[:, channel: channel + 1, :, :] * cos,
                                       spectrogram[:, channel: channel + 1, :, :] * sin, length))
        output = torch.stack(wav_list, dim=1)
        return output

    def wav_to_mag_phase_subband_spectrogram(self, input):
        """
        :param input:
        :param eps:
        :return:
            loss = torch.nn.L1Loss()
            model = FDomainHelper(subband=4)
            data = torch.randn((3,1, 44100*3))

            sps, coss, sins = model.wav_to_mag_phase_subband_spectrogram(data)
            wav = model.mag_phase_subband_spectrogram_to_wav(sps,coss,sins,44100*3//4)

            print(loss(data,wav))
            print(torch.max(torch.abs(data-wav)))

        """
        length, pad = input.size()[-1], 0
        while ((length + pad) % self.subband != 0):  pad += 1
        input = F.pad(input, (0, pad))
        if(self.subband > 1):
            subwav = self.qmf.analysis(input) # [batchsize, subband*channels, samples]
        else:
            subwav = input
        sps, coss, sins = self.wav_to_spectrogram_phase(subwav)
        return sps,coss,sins

    def mag_phase_subband_spectrogram_to_wav(self, sps,coss,sins, length):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        subwav = self.spectrogram_phase_to_wav(sps,coss,sins, ceil(length / self.subband) + 64 // self.subband)
        if(self.subband > 1): data = self.qmf.synthesis(subwav)
        else: data = subwav
        return data[...,:length]

    def wav_to_subband(self, input):
        """
        :param input:
        :param eps:
        :return:
            loss = torch.nn.L1Loss()
            model = FDomainHelper(subband=4)
            data = torch.randn((3,1, 44100*3))

            sps, coss, sins = model.wav_to_mag_phase_subband_spectrogram(data)
            wav = model.mag_phase_subband_spectrogram_to_wav(sps,coss,sins,44100*3//4)

            print(loss(data,wav))
            print(torch.max(torch.abs(data-wav)))

        """
        length, pad = input.size()[-1], 0
        while ((length + pad) % self.subband != 0):  pad += 1
        input = F.pad(input, (0, pad))
        if(self.subband > 1):
            subwav = self.qmf.analysis(input) # [batchsize, subband*channels, samples]
        else:
            subwav = input
        return subwav

    def subband_to_wav(self, subwav, length):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        if(self.subband > 1): data = self.qmf.synthesis(subwav)
        else: data = subwav
        return data[...,:length]


    # todo the following code is not bug free!
    def wav_to_complex_spectrogram(self, input):
        # [batchsize , channels, samples]
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        res = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            res.append(self._complex_spectrogram(input[:, channel, :]))
        return torch.cat(res,dim=1)

    def complex_spectrogram_to_wav(self, input, length=None):
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        # return  [batchsize, channels, samples]
        channels = input.size()[1] // 2
        wavs = []
        for i in range(channels):
            wavs.append(self._reverse_complex_spectrogram(input[:, 2 * i:2 * i + 2, ...], length = length))
            wavs[-1] = wavs[-1].unsqueeze(1)
        return torch.cat(wavs,dim=1)

    def wav_to_complex_subband_spectrogram(self, input):
        # [batchsize, channels, samples]
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        length, pad = input.size()[-1], 0
        while ((length + pad) % self.subband != 0):  pad += 1
        input = F.pad(input, (0, pad))
        if (self.subband > 1):
            subwav = self.qmf.analysis(input)  # [batchsize, subband*channels, samples]
        else:
            subwav = input
        subspec = self.wav_to_complex_spectrogram(subwav)
        return subspec

    def complex_subband_spectrogram_to_wav(self, input, length):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        subwav = self.complex_spectrogram_to_wav(input, length = ceil(length / self.subband) + 64 // self.subband)
        if (self.subband > 1):
            data = self.qmf.synthesis(subwav)
        else:
            data = subwav
        return data[...,:length]

def test():
    loss = torch.nn.L1Loss()


    print("Convert time sequence into subband time samples.")
    for SUB in [2, 4, 8]:
        for length in range(1, 5):
            for channel in [1, 2, 3]:
                model = SubbandDSP(subband=SUB)
                data = torch.randn((3, channel, 44100 * length))
                subwav = model.wav_to_subband(data)
                wav = model.subband_to_wav(subwav, 44100 * length)
                print(SUB, "subbands;", channel, "channels;", str(length) + "s audio;" " reconstruction l1loss: ",
                      float(loss(data, wav)), "; relative loss: ",
                      "{:.5}".format(float(loss(data, wav) / torch.mean(torch.abs(data))) * 100) + "%")

    print("Convert time sequence into subband spectrograms.")
    for SUB in [2, 4, 8]:
        for length in range(1, 5):
            for channel in [1, 2, 3]:
                model = SubbandDSP(subband=SUB)
                data = torch.randn((3, channel, 44100 * length))
                sps = model.wav_to_complex_subband_spectrogram(data)
                wav = model.complex_subband_spectrogram_to_wav(sps, 44100 * length)
                print(SUB, "subbands;", channel, "channels;", str(length) + "s audio;" " reconstruction l1loss: ",
                      float(loss(data, wav)), "; relative loss: ",
                      "{:.5}".format(float(loss(data, wav) / torch.mean(torch.abs(data))) * 100) + "%")

    print("Convert time sequence into subband complex as channel")
    for SUB in [2, 4, 8]:
        for length in range(1, 5):
            for channel in [1, 2, 3]:
                model = SubbandDSP(subband=SUB)
                data = torch.randn((3, channel, 44100 * length))
                sps, coss, sins = model.wav_to_mag_phase_subband_spectrogram(data)
                wav = model.mag_phase_subband_spectrogram_to_wav(sps, coss, sins, 44100 * length)
                print(SUB, "subbands;", channel, "channels;", str(length) + "s audio;" " reconstruction l1loss: ",
                      float(loss(data, wav)), "; relative loss: ",
                      "{:.5}".format(float(loss(data, wav) / torch.mean(torch.abs(data))) * 100) + "%")