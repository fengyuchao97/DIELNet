# Rethinking Semantic-level Building Change Detection: Ensemble Learning and Dynamic Interaction

Here, we provide the pytorch implementation of the paper for semantic-level BCD: Rethinking Semantic-level Building Change Detection: Ensemble Learning and Dynamic Interaction.

## 1. Environment setup
This code has been tested on on the workstation with Intel Xeon CPU E5-2690 v4 cores and GPUs of NVIDIA TITAN V with a single 12G of video memory, Python 3.6, pytorch 1.9, CUDA 10.0, cuDNN 7.6.

## 2. Download the datesets:
* LEVIR-CD+:
[LEVIR-CD+](https://pan.baidu.com/s/1wxr9GoI8XrUuHCdWO-SOTA?pwd=nr6u)
* LEVIR-CD:
[LEVIR-CD](https://pan.baidu.com/s/10FchKHUynowsOozDIuXT3w?pwd=8rzq)
* WHU-CD:
[WHU-CD](https://pan.baidu.com/s/1XnSW_z84r7nIq5WEhcPMDg?pwd=cj4s)
* GZ-CD:
[GZ-CD](https://pan.baidu.com/s/1otPrEKsGYjtaaOfoG1CFgQ?pwd=hdiv)
* Inria-CD:
[Inria-CD](https://pan.baidu.com/s/1XbJ9pQKvqCndhyxJ77eZow?pwd=38vy)

and put them into data directory 'All'. Or you can just download our prepared datasets.

* All:
[All]()


## 3. Train

    python train_cd.py
    
## 4. Test

    python eval_cd.py


## 5. Cite
If you use our method in your work please cite our paper:

