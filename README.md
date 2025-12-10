## Reduction kernel in CUDA
#### This program was written for [HPCGames](https://hpcgame.pku.edu.cn/), based on [NVIDIA Guides](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf). Modified to fit modern GPU architectures.
<hr>

## ❔What does the program do?
### This program generates a batch of random float value based on _gaussian distribution_, then implements CUDA API for reduction. Can be used for benchmarking performance across different implementations.
## ✨Update
### The more modern approach ___shfl_down_sync()_ is implemented in the current version, replacing the previous warp reduction kernel.
### The current kernel  is now using _thread coarsing_, outperforming the previous kernel  by 144.3% on large test sets. 
## ‼️Notice
### As was tested, the kernel performs the best when the number of _blocks_ is integer times the number of _Stream Processors_ on the GPU. You may adjust N_BLOCK_NUM in the source file for further testing.
