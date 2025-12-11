## CUDA Reduction kernel
#### This program was written for [HPCGames](https://hpcgame.pku.edu.cn/), based on [NVIDIA Guides](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf). Modified to fit modern GPU architectures.
<hr>

### ❔What does the program do?
 This program generates a batch of random float value based on _gaussian distribution_, then implements CUDA API for reduction. Can be used for benchmarking performance across different implementations.
### 🔨Build and run
You may use the CMakeLists.txt for buildfile generation.<br>
It is adviced to change compiler flags based on your GPU's [Compute Capability](https://developer.nvidia.com/cuda-gpus) (e.g. -arch=sm_86 if using 3090, CC = 8.6)<br>
To build project,
```
$ chmod 777 build.sh
$ ./build.sh
```
To test run code,
```
$ chmod 777 run.sh
$ ./run.sh
```
### 🧭Results
#### Test results on 3090
```
+--------
cuReduce.cu
101694.885373
101694.885373
101694.885373
101694.885373
len: 1024000 , time: 232.723 us
Result correct! diff is 0.0078125
+--------
cuReductionOptimized.cu
101694.885373
101694.885373
101694.885373
101694.885373
len: 1024000 , time: 210.921 us
Result correct! diff is 0.015625
+--------
cuReduce.cu
818698.212473
818698.212473
818698.212473
818698.212473
len: 8192000 , time: 336.406 us
Result correct! diff is 0.0625
+--------
cuReductionOptimized.cu
818698.212473
818698.212473
818698.212473
818698.212473
len: 8192000 , time: 248.4 us
Result correct! diff is 0.0625
+--------
cuReduce.cu
4105693.349531
4105693.349531
4105693.349531
4105693.349531
len: 40960000 , time: 686.267 us
Result correct! diff is 0.5
+--------
cuReductionOptimized.cu
4105693.349531
4105693.349531
4105693.349531
4105693.349531
len: 40960000 , time: 496.29 us
Result correct! diff is 0.25
+--------
cuReduce.cu
10251040.582175
10251040.582175
10251040.582175
10251040.582175
len: 102400000 , time: 1462.32 us
Result correct! diff is 2
+--------
cuReductionOptimized.cu
10251040.582175
10251040.582175
10251040.582175
10251040.582175
len: 102400000 , time: 959.988 us
Result correct! diff is 1
```
### ✨Update
 The more modern approach ___shfl_down_sync()_ is implemented in the current version, replacing the previous warp reduction kernel.<br>
 The current kernel  is now using _thread coarsing_, significantly outperforming the previous kernel on large test sets.
### ‼️Notice
As was tested, the kernel performs the best when the number of _blocks_ is integer times the number of _Stream Processors_ on the GPU. You may adjust N_BLOCK_NUM in the source file for further testing.
