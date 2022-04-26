## Sparse Logits Suffice to Fail Knowledge Distillation

["Sparse Logits Suffice to Fail Knowledge Distillation"](https://openreview.net/pdf?id=BxZgduuNDl5)     

Haoyu Ma, Yifan Huang, Hao Tang, Chenyu You, Deying Kong, Xiaohui Xie  
In ICLR 2022 Workshop [PAIR^2Struct](https://pair2struct-workshop.github.io/)
   


## Prerequisite
We use Pytorch 1.7.1, and CUDA 10.1. You can install them with  It should also be applicable to other Pytorch and CUDA versions.  


Then install other packages by
~~~
pip install -r requirements.txt
~~~

## Usage 


### Teacher networks 

##### Train a normal teacher network   

For example, normally train a ResNet18 on CIFAR-10  
~~~
python train_scratch.py --save_path experiments/CIFAR100/baseline/resnet18
~~~
After finishing training, you will get `training.log`, `best_model.tar` in that directory.  
   


### Student networks 
##### Knowledge Distillation for Student networks 

Train a ShuffleNet-v2 distilling from a Stingy ResNet18 
~~~
python train_kd_stingy.py --save_path experiments/CIFAR100/kd_stingy_resnet18/shufflenetv2
~~~


## Citation
If you find our code helps your research, please cite the paper:

~~~
@inproceedings{
ma2022sparse,
title={Sparse Logits Suffice to Fail Knowledge Distillation},
author={Haoyu Ma and Yifan Huang and Hao Tang and Chenyu You and Deying Kong and Xiaohui Xie},
booktitle={ICLR 2022 Workshop on PAIR{\textasciicircum}2Struct: Privacy, Accountability, Interpretability, Robustness, Reasoning on Structured Data},
year={2022},
url={https://openreview.net/forum?id=BxZgduuNDl5}
}
~~~











