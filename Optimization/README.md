
# EDGY
```
We evaluate EDGY's on-device performance, and explore optimization techniques, including model pruning 
and quantization, to enable private, accurate and efficient representation learning on resource-constrained devices.
```
## Introduction
Inference efficiency is a significant challenge when deploying DL models at the edge given restrictions on processing, memory, and in some cases power consumption. To address this challenge, we focus on a variety of techniques that involve reducing model parameters with pruning and/or reducing representational precision with quantization to support efficient inference at the edge. 

## Optimization Tools
* Automatic Mixed Precision [AMP](https://nvidia.github.io/apex/amp.html), a PyTorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training. 
* Neural Network Compression Framework [NNCF](https://github.com/openvinotoolkit/nncf), a framework for neural network compression with fine-tuning built on the top of PyTorch framework, to experiment with different compression techniques. It supports various compression algorithms including quantization, pruning, and sparsity applied during the model fine-tuning process to achieve better compression parameters and accuracy. <br />
**Note: you only need to install these packages and add them to your training script

## Implementation
We elaborate in the following:
* (1) Low-precision Training (AMP).<br />
Apex offers four optimization levels, as follows: O0 (pure FP32 training), O1 (Conservative Mixed Precision), O2 (Fast Mixed Precision), and O3 (FP16 training). Initialize your model with "amp.initialize" and set the optimization level flag to O1. In the optimization level O1, it uses dynamic loss scaling, and all Torch functions and Tensor methods cast their inputs according to a whitelist-blacklist model. The whitelist operations (i.e., Tensor Core-friendly ones, like convolutions) are performed in FP16, while the blacklist operations that benefit from FP32 precision (e.g., softmax) are performed in FP32. The following three lines need to be added to the main training script:

```
# Step 1: Initialization
opt_level = 'O1'
[encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level=opt_level)

# Step 2: Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...

# Step 3: Save checkpoint
checkpoint_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
torch.save(checkpoint_state, checkpoint_path)
...
```
<br />

* (2) Pruning and Sparsity (NNCF).<br />
* (3) Quantization (NNCF).<br />
The overall compression procedures (i.e., Pruning and Quantization) can be summarized as: loading a JSON configuration script which contains NNCF-specific parameters determining the compression to be applied to the model, and then passing the FP model along with the configuration script to the "nncf.create\_compressed\_model" function. This function returns a wrapped model ready for compression and fine-tuning, and an additional object to allow further control of the compression during the fine-tuning process, as shown in the Figure. Fine-tuning is a necessary step in some cases to recover the ability to generalize that may have been damaged by the model optimization techniques. 
![GitHub Logo](/images/ModelOptimization_Pipeline.png)
<br />
```
# Step 1: Create an NNCF configuration file
You can find diffrent configration files examples in JSON_configration folder to setup the parameters of compression to be applied to your model.

# Step 2: Add the imports required for NNCF
import torch
import nncf  # Important - should be imported directly after torch
from nncf import create_compressed_model, NNCFConfig, register_default_init_args

# Step 3: Load the NNCF JSON configuration file that you prepared during Step 1
nncf_config = NNCFConfig.from_json("nncf_config.json")  

# Step 4: Provide data loaders for compression algorithm initialization, if necessary (i.e., Quantization Implementation)
nncf_config = register_default_init_args(nncf_config, train_loader, loss_criterion)

# Step 5: # Apply the specified compression algorithms to the model
comp_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Step 6: In the training loop, 
          * After inferring the model, take a compression loss and add it (using the + operator) to the common loss, 
          compression_loss = compression_ctrl.loss()
          loss = common_loss + compression_loss
          
          * Call the scheduler step() after each training iteration:
          compression_ctrl.scheduler.step()
          
          * Call the scheduler epoch_step() after each training epoch:
          compression_ctrl.scheduler.epoch_step()

```
<br />
** Note: ![for more detailes](https://github.com/openvinotoolkit/nncf/blob/develop/docs/Usage.md), this is a step-by-step tutorial on how to integrate it.
