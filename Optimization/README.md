
# EDGY
```
We evaluate EDGY's on-device performance, and explore optimization techniques, including model pruning 
and quantization, to enable private, accurate and efficient representation learning on resource-constrained devices.
```
## Introduction
Inference efficiency is a significant challenge when deploying DL models at the edge given restrictions on processing, memory, and in some cases power consumption. To address this challenge, we focus on a variety of techniques that involve reducing model parameters with pruning and/or reducing representational precision with quantization to support efficient inference at the edge. 

## Optimization Tools
* Automatic Mixed Precision [AMP](https://nvidia.github.io/apex/amp.html), a PyTorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training. 
* Neural Network Compression Framework [NNCF](https://github.com/openvinotoolkit/nncf), a framework for neural network compression with fine-tuning built on the top of PyTorch framework, to experiment with different compression techniques. It supports various compression algorithms including quantization, pruning, and sparsity applied during the model fine-tuning process to achieve better compression parameters and accuracy. 


## Implementation
We elaborate in the following:
* (1) Low-precision Training (using AMP).
Apex offers four optimization levels, as follows: O0 (pure FP32 training), O1 (Conservative Mixed Precision), O2 (Fast Mixed Precision), and O3 (FP16 training). We initialize our model with "amp.initialize" and set the optimization level flag to O1. In the optimization level O1, it uses dynamic loss scaling, and all Torch functions and Tensor methods cast their inputs according to a whitelist-blacklist model. The whitelist operations (i.e., Tensor Core-friendly ones, like convolutions) are performed in FP16, while the blacklist operations that benefit from FP32 precision (e.g., softmax) are performed in FP32.  

```
We evaluate EDGY's on-device performance, and explore optimization techniques, including model pruning 
and quantization, to enable private, accurate and efficient representation learning on resource-constrained devices.
```
* (2) Pruning and Sparsity (using NNCF).
* (3) Quantization (using NNCF).
The overall compression procedures (i.e., Pruning and Quantization) can be summarized as: loading a JSON configuration script which contains NNCF-specific parameters determining the compression to be applied to the model, and then passing the FP model along with the configuration script to the "nncf.create\_compressed\_model" function. This function returns a wrapped model ready for compression and fine-tuning, and an additional object to allow further control of the compression during the fine-tuning process, as shown in the Figure. Fine-tuning is a necessary step in some cases to recover the ability to generalize that may have been damaged by the model optimization techniques. 
![GitHub Logo](/images/ModelOptimization_Pipeline.png)
```
We evaluate EDGY's on-device performance, and explore optimization techniques, including model pruning 
and quantization, to enable private, accurate and efficient representation learning on resource-constrained devices.
```
