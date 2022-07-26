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
import numpy as np
from torchsubband.pqmf import PQMF
import torch.nn.functional as F
from math import ceil
from torch import nn

class SubbandDSP(nn.Module):
    def __init__(self,
             subband = 2,
             window_size=2048,
             hop_size=441,
             ):
        """
        Args:
            subband: int, [1,2,4,8]. The subband number you wanna divide. 'subbband==1' means do not need subband.
            window_size: stft parameter
            hop_size: stft parameter
        """
        super(SubbandDSP, self).__init__()
        center = True
        pad_mode = 'reflect'
        window = 'hann'
        freeze_parameters = True
        self.subband = subband
        self.stft = STFT(n_fft=window_size // self.subband, hop_length=hop_size // self.subband,
                         win_length=window_size // self.subband, window=window, center=center,
                         pad_mode=pad_mode, freeze_parameters=freeze_parameters)
        self.istft = ISTFT(n_fft=window_size // self.subband, hop_length=hop_size // self.subband,
                           win_length=window_size // self.subband, window=window, center=center,
                           pad_mode=pad_mode, freeze_parameters=freeze_parameters)   
        if(subband > 1):
            self.qmf = PQMF(subband, 64)

    def wav_to_wavegram(self, input, power_of_two=5):
        """
        Convert input waveform into wavegram
        Args:
            input: tensor, (batch_size, channels_num, samples)
        Returns:
            tensor, 
        """
        assert self.subband == 2, "Error: To use wavegram feature, you need to set the subband number to 2."
    
        length = []
        for _ in range(power_of_two):
            length.append(input.shape[-1])
            input = self.wav_to_sub(input)
        return input, length
    
    def wavegram_to_wav(self, wavegram, length, power_of_two=5):
        """
        Convert wavegram into waveform
        Args:
            input: tensor, 
        Returns:
            tensor, (batch_size, channels_num, samples)
        """
        for i in range(power_of_two):
            wavegram = self.sub_to_wav(wavegram, length=length[(-(i+1))])
        return wavegram

    def wav_to_sub(self, input):
        """
        Convert input waveform into several subband signals
        Args:
            input: tensor, (batch_size, channels_num, samples)
        Returns:
            tensor, (batch_size, channels_num * subbandnum, ceil(samples / subbandnum))
        """
        length, pad = input.size()[-1], 0
        while ((length + pad) % self.subband != 0):     pad += 1
        input = F.pad(input, (0, pad))
        if(self.subband > 1):
            subwav = self.qmf.analysis(input) # [batchsize, subband*channels, samples]
        else:
            subwav = input
        return subwav

    def sub_to_wav(self, subwav, length):
        """
        The reverse function of wav_to_subband.
        Args:
            subwav: tensor, (batch_size, channels_num * subband_nums, ceil(samples / subbandnum))
            length: int, expect sample length of the output tensor

        Returns:
            tensor, (batch_size, channels_num, samples)
        """
        if(self.subband > 1): data = self.qmf.synthesis(subwav)
        else: data = subwav
        return data[...,:length]

    def wav_to_spectrogram_phase(self, input):
        """
        Convert input waveform to magnitude spectrogram and phases.
        Args:
            input: (batch_size, channels_num, samples)
        Returns:
            magnitude spectrogram (batch_size, channels_num, time_steps, freq_bins),
            phase angle cos: (batch_size, channels_num, time_steps, freq_bins)
            phase angle sin: (batch_size, channels_num, time_steps, freq_bins)
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
        """
        The reverse function of wav_to_spectrogram_phase. Convert magnitude spectrogram and phase to waveform.
        Args:
            sps: tensor, magnitude spectrogram, (batch_size, channels_num, time_steps, freq_bins),
            coss: tensor, phase angle, (batch_size, channels_num, time_steps, freq_bins)
            sins: tensor, phase angle, (batch_size, channels_num, time_steps, freq_bins)
            length: int, expect sample length of the output tensor
        Returns:
            output: tensor, (batch_size, channels_num, samples)
        """
        channels_num = sps.size()[1]
        res = []
        for i in range(channels_num):
            res.append(self.istft(sps[:,i:i+1,...] * coss[:,i:i+1,...], sps[:,i:i+1,...] * sins[:,i:i+1,...], length))
            res[-1] = res[-1].unsqueeze(1)
        return torch.cat(res,dim=1)

    def wav_to_mag_phase_sub_spec(self, input):
        """
        Convert the input waveform to its subband spectrograms, which are concatenated in the channel dimension.
        Args:
            input: (batch_size, channels_num, samples)
        Returns:
                magnitude spectrogram (batch_size, channels_num * subband_num, time_steps, freq_bins // subband_num),
                coss: (batch_size, channels_num * subband_num, time_steps, freq_bins // subband_num)
                sins: (batch_size, channels_num * subband_num, time_steps, freq_bins // subband_num)
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

    def mag_phase_sub_spec_to_wav(self, sps, coss, sins, length):
        """
        The reverse functino of wav_to_mag_phase_subband_spectrogram. Convert subband magnutde spectrogram and its subband phases into fullband waveform.
        Args:
            sps: tensor, magnitude spectrogram (batch_size, channels_num * subband_num, time_steps, freq_bins // subband_num),
            coss: tensor, (batch_size, channels_num * subband_num, time_steps, freq_bins // subband_num)
            sins: tensor, (batch_size, channels_num * subband_num, time_steps, freq_bins // subband_num)
            length: int, expect sample length of the output tensor
        Returns:
             tensor, (batch_size, channels_num, samples)
        """
        subwav = self.spectrogram_phase_to_wav(sps,coss,sins, ceil(length / self.subband) + 64 // self.subband)
        if(self.subband > 1): data = self.qmf.synthesis(subwav)
        else: data = subwav
        return data[...,:length]

    def wav_to_complex_sub_spec(self, input):
        """
        Convert waveform in each channel to several complex subband spectrogram. The real and imaginary parts are stored separately in different channels.
        Args:
            input: tensor, (batch_size, channels_num, samples)

        Returns:
            tensor, complex as channel spectrogram, (batch_size, 2 * channels_num * subband_num, time_steps, freq_bins // subband_num),
        """
        length, pad = input.size()[-1], 0
        while ((length + pad) % self.subband != 0):  pad += 1
        input = F.pad(input, (0, pad))
        if (self.subband > 1):
            subwav = self.qmf.analysis(input)  # [batchsize, subband*channels, samples]
        else:
            subwav = input
        subspec = self._wav_to_complex_spectrogram(subwav)
        return subspec

    def complex_sub_spec_to_wav(self, sps, length):
        """
        The reverse function of wav_to_complex_subband_spectrogram. Convert complex spectrogram into waveform.
        Args:
            sps: tensor, complex as channel spectrogram, (batch_size, 2 * channels_num * subband_num, time_steps, freq_bins // subband_num),
            length: int, expect sample length of the output tensor

        Returns:
            (batch_size, channels_num, samples)
        """
        subwav = self._complex_spectrogram_to_wav(sps, length =ceil(length / self.subband) + 64 // self.subband)
        if (self.subband > 1):
            data = self.qmf.synthesis(subwav)
        else:
            data = subwav
        return data[...,:length]

    def _complex_spectrogram(self, input):
        real, imag = self.stft(input)
        return torch.cat([real, imag], dim=1)

    def _reverse_complex_spectrogram(self, input, length=None):
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

    def _wav_to_spectrogram(self, input):
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

    def _spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.
        Args:
          input: (batch_size, channels_num, segment_samples)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          tensor, (batch_size, channels_num, segment_samples)
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

    def _wav_to_complex_spectrogram(self, input):
        res = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            res.append(self._complex_spectrogram(input[:, channel, :]))
        return torch.cat(res,dim=1)

    def _complex_spectrogram_to_wav(self, input, length=None):
        channels = input.size()[1] // 2
        wavs = []
        for i in range(channels):
            wavs.append(self._reverse_complex_spectrogram(input[:, 2 * i:2 * i + 2, ...], length = length))
            wavs[-1] = wavs[-1].unsqueeze(1)
        return torch.cat(wavs,dim=1)

def test():
    # from torchsubband import SubbandDSP

    loss = torch.nn.L1Loss()

    print("Convert time sequence into subband time samples.")
    for SUB in [1, 2, 4, 8]:
        for length in range(1, 5):
            for channel in [1, 2, 3]:
                model = SubbandDSP(subband=SUB)
                data = torch.randn((3, channel, 44100 * length))
                subwav = model.wav_to_sub(data)
                wav = model.sub_to_wav(subwav, 44100 * length)
                print(SUB, "subbands;", channel, "channels;", str(length) + "s audio;" " reconstruction l1loss: ",
                      float(loss(data, wav)), "; relative loss: ",
                      "{:.5}".format(float(loss(data, wav) / torch.mean(torch.abs(data))) * 100) + "%")

    print("Convert time sequence into subband complex spectrogram")
    for SUB in [1, 2, 4, 8]:
        for length in range(1, 5):
            for channel in [1, 2, 3]:
                model = SubbandDSP(subband=SUB)
                data = torch.randn((3, channel, 44100 * length))
                sps = model.wav_to_complex_sub_spec(data)
                wav = model.complex_sub_spec_to_wav(sps, 44100 * length)
                print(SUB, "subbands;", channel, "channels;", str(length) + "s audio;" " reconstruction l1loss: ",
                      float(loss(data, wav)), "; relative loss: ",
                      "{:.5}".format(float(loss(data, wav) / torch.mean(torch.abs(data))) * 100) + "%")

    print("Convert time sequence into subband magnitude spectrograms and phase.")
    for SUB in [1, 2, 4, 8]:
        for length in range(1, 5):
            for channel in [1, 2, 3]:
                model = SubbandDSP(subband=SUB)
                data = torch.randn((3, channel, 44100 * length))
                sps, coss, sins = model.wav_to_mag_phase_sub_spec(data)
                wav = model.mag_phase_sub_spec_to_wav(sps, coss, sins, 44100 * length)
                print(SUB, "subbands;", channel, "channels;", str(length) + "s audio;" " reconstruction l1loss: ",
                      float(loss(data, wav)), "; relative loss: ",
                      "{:.5}".format(float(loss(data, wav) / torch.mean(torch.abs(data))) * 100) + "%")

if __name__ == "__main__":
    dsp = SubbandDSP()
    data = torch.randn((3, 2, 441000))
    wavegram, length = dsp.wav_to_wavegram(data, power_of_two=10)
    print(wavegram.size())
    waveform = dsp.wavegram_to_wav(wavegram, length, power_of_two=10)
    print(torch.mean(torch.abs(data-waveform)))
