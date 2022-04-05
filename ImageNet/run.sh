


# R18 KD from DenseNet121 (Normal) 
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train_kd.py --arch resnet18 --arch_t densenet121 --save_dir ckpt_resnet18_kd --apex 1 --gpu 0 1 2 3 > log18kd 2>&1 &

# R18 KD from DenseNet121 (Stingy) 
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train_kd_stingy.py --arch resnet18 --arch_t densenet121 --save_dir ckpt_resnet18_kd_stingy --apex 1 --gpu 0 1 2 3 --temperature 20 --img_dir XXXX > log18kd 2>&1 &


