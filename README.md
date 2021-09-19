# torchsubband

This's a package for subband decomposition 

## Reconstruction loss

We tried a 60 seconds long audio to do subband decomposition and calculated the following metrics.

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

