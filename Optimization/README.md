
# EDGY
```
We evaluate EDGY's on-device performance, and explore optimization techniques, including model pruning and quantization, to enable private, accurate and efficient representation learning on resource-constrained devices.
```
## Introduction
Inference efficiency is a significant challenge when deploying DL models at the edge given restrictions on processing, memory, and in some cases power consumption. To address this challenge, we focus on a variety of techniques that involve reducing model parameters with pruning and/or reducing representational precision with quantization to support efficient inference at the edge. 

We elaborate in the following:
*(1) Low-precision Training.
*(2) Pruning and Sparsity.
*(3) Quantization.

We use the Neural Network Compression Framework [NNCF](https://github.com/openvinotoolkit/nncf), a framework for neural network compression with fine-tuning built on the top of PyTorch framework, to experiment with different compression techniques. 
It supports various compression algorithms including quantization, pruning, and sparsity applied during the model fine-tuning process to achieve better compression parameters and accuracy. 
![GitHub Logo](EDGY/images/ModelOptimization_Pipeline.pdf)
The overall compression procedures can be summarized as: loading a JSON configuration script which contains NNCF-specific parameters determining the compression to be applied to the model, and then passing the FP model along with the configuration script to the "nncf.create\_compressed\_model" function. This function returns a wrapped model ready for compression and fine-tuning, and an additional object to allow further control of the compression during the fine-tuning process, as in Figure. Fine-tuning is a necessary step in some cases to recover the ability to generalize that may have been damaged by the model optimization techniques. 

