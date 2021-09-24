# torchsubband

This's a package for subband decomposition. 

It can transform waveform into three kinds of revertable subband feature representations, which are potentially useful features for music source separation or similar tasks.

[![488zR0.png](https://z3.ax1x.com/2021/09/19/488zR0.png)](https://imgtu.com/i/488zR0)

## Usage

Installation
```shell
pip install torchsubband
```

A simple example: 

```python
from torchsubband import SubbandDSP
import torch

# nn.Module
model = SubbandDSP(subband=2) # You can choose 1,2,4, or 8 
batchsize=3 # any int number
channel=1 # any int number
length = 44100*2 # any int number
input = torch.randn((batchsize,channel,length))

# Get subband waveform
subwav = model.wav_to_sub(input)
reconstruct_1 = model.sub_to_wav(subwav,length=length)

# Get subband magnitude spectrogram
sub_spec,cos,sin = model.wav_to_mag_phase_sub_spec(input)
reconstruct_2 = model.mag_phase_sub_spec_to_wav(sub_spec,cos,sin,length=length)

# Get subband complex spectrogram
sub_complex_spec = model.wav_to_complex_sub_spec(input)
reconstruct_3 = model.complex_sub_spec_to_wav(sub_complex_spec,length=length)
```

## Reconstruction loss

The following table shows the reconstruction quality. We tried a set of audio to conduct subband decomposition and reconstruction.


| Subbands |  L1loss   | PESQ  | SiSDR|
| :----: | :----: | :----: | :----:
| 2 | 1e-6  | 4.64 | 61.8 |
| 4 | 1e-6  | 4.64 | 58.9 |
| 8 | 5e-5  | 4.64 | 58.2 |

You can also test this program by training the following test script. It will give you some evaluation output.

```python
from torchsubband import test
test()
```

## Citation

If you find our code useful for your research, please consider citing:

>    @inproceedings{Liu2020,   
>      author={Haohe Liu and Lei Xie and Jian Wu and Geng Yang},   
>      title={{Channel-Wise Subband Input for Better Voice and Accompaniment Separation on High Resolution Music}},   
>      year=2020,   
>      booktitle={Proc. Interspeech 2020},   
>      pages={1241--1245},   
>      doi={10.21437/Interspeech.2020-2555},   
>      url={http://dx.doi.org/10.21437/Interspeech.2020-2555}   
>    }.
