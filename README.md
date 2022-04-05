## Stingy Teacher


## Prerequisite
We use Pytorch 1.7.1, and CUDA 10.1. You can install them with  It should also be applicable to other Pytorch and CUDA versions.  


Then install other packages by
~~~
pip install -r requirements.txt
~~~

## Usage 


### Teacher networks 

##### Step 1: Train a normal teacher network   

For example, normally train a ResNet18 on CIFAR-10  
~~~
python train_scratch.py --save_path experiments/CIFAR100/baseline/resnet18
~~~
After finishing training, you will get `training.log`, `best_model.tar` in that directory.  
   


### Step 2: Knowledge Distillation for Student networks 

* Train a ShuffleNet-v2 distilling from a Stingy ResNet18 
~~~
python train_kd_stingy.py --save_path experiments/CIFAR100/kd_stingy_resnet18/shufflenetv2
~~~
