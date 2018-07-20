
CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds norm  --keep_prob 0.5  --log_dir log &
CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds norm  --keep_prob 0.5 --no_shuffle  --log_dir log 

#CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds ply  --keep_prob 0.5  --log_dir log &
#CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds ply  --keep_prob 0.5 --no_shuffle  --log_dir log 
#
#CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds norm  --keep_prob 0.5 --num_point 4096  --log_dir log &
#CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds norm  --keep_prob 0.5 --num_point 4096 --no_shuffle  --log_dir log 
#
#CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug A  --single_scale  --ds norm  --keep_prob 0.5  --log_dir log &
#CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug A  --single_scale  --ds norm  --keep_prob 0.5 --no_shuffle  --log_dir log 
#
#CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug A  --single_scale  --ds norm  --keep_prob 0.2  --log_dir log &
#CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug A  --single_scale  --ds norm  --keep_prob 0.2 --no_shuffle  --log_dir log 
