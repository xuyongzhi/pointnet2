#!/bin/bash
#PBS -q gpu
#PBS -l walltime=20:00:00
#PBS -l mem=16GB
#PBS -l jobfs=0GB
#PBS -l ngpus=2
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.8-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
##PBS -M y.xu@unsw.edu.au
##PBS -m abe

module load  tensorflow/1.8-cudnn7.1-python2.7
module list
 

#------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds ply  --keep_prob 0.5  --log_dir log &
CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds ply  --keep_prob 0.5 --no_shuffle  --log_dir log 

CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds norm  --keep_prob 0.5 --num_point 4096  --log_dir log &
CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug N  --single_scale  --ds norm  --keep_prob 0.5 --num_point 4096 --no_shuffle  --log_dir log 

CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug A  --single_scale  --ds norm  --keep_prob 0.5  --log_dir log &
CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug A  --single_scale  --ds norm  --keep_prob 0.5 --no_shuffle  --log_dir log 

CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py  --num_gpus 1  --aug A  --single_scale  --ds norm  --keep_prob 0.2  --log_dir log &
CUDA_VISIBLE_DEVICES=1 python train_multi_gpu.py  --num_gpus 1  --aug A  --single_scale  --ds norm  --keep_prob 0.2 --no_shuffle  --log_dir log 
