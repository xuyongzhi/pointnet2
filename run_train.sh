#CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py --num_gpus 2 --normal

CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py --num_gpus 1 --optimizer momentum --learning_rate 0.01 --log_dir log_mom_lr0d01 --max_epoch 121
