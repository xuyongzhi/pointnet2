#CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py --num_gpus 2 --normal

CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py --num_gpus 2  --aug No --log_dir log_NoAug
