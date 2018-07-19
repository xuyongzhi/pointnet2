#CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py --num_gpus 2  --aug No --log_dir log_NoAug

CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py --num_gpus 1  --aug No  --single_scale  --num_point 2048 --log_dir log_SingeS_NoAug_WithNorm2048

#CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py --num_gpus 1  --aug No  --single_scale --keep_prob 1.1  --log_dir log_NoAug_SingeS_NoDrop
