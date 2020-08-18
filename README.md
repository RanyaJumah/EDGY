# EDGY
```
This repo provides the code for 
"Privacy-preserving Voice Analysis via Disentangled Representations" paper.
```
## Introduction
Voice User Interfaces (VUIs) are increasingly popular and built into smartphones, home assistants, and Internet of Things (IoT) devices. Despite offering an always-on convenient user experience, VUIs raise new security and privacy concerns for their users. 
In this paper, we focus on attribute inference attacks in the speech domain, As shown in the figure below, demonstrating the potential for an attacker to accurately infer a target user's sensitive and private attributes (e.g. their emotion, sex, or health status) from deep acoustic models. To defend against this class of attacks, we design, implement, and evaluate a user-configurable, privacy-aware framework for optimizing speech-related data sharing mechanisms.
![GitHub Logo](/images/Potential_Attacks.png)


## Requirements
* Ensure you have Python 3 and PyTorch 1.4 or greater
* Install [NVIDIA/apex](https://github.com/NVIDIA/apex) for mixed precision training 
* Install all package dependencies 


## Download Dataset and Training
```
Details under **Training Folder**
```

## Usage
```
python DDF.py checkpoint=path/to/checkpoint in_dir=path/to/wavs out_dir=path/to/out_dir synthesis_list=path/to/synthesis_list dataset=[???]
```
To generate different output:

Privacy_preference = [Low, Moderate, High]
Output_type = [Recording, Embedding]

## Cite
```
If you find this work useful, please cite us.
@misc{aloufi2020privacypreserving,
    title={Privacy-preserving Voice Analysis via Disentangled Representations},
    author={Ranya Aloufi and Hamed Haddadi and David Boyle},
    year={2020},
    eprint={2007.15064},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

## Contributing
```
This project welcomes contributions and suggestions. 
Please contact **ra6018@imperial.ac.uk** with any additional questions or comments.
```


