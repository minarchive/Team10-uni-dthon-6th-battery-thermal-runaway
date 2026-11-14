#!/bin/bash
#SBATCH --job-name  exp_retr
#SBATCH --time      12:00:00
#SBATCH -c          6
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    exp_retrieval_gpu.log

cd "/home/nxcolelxu/unid/uni_d_baseline_v7 (1)/uni_d_baseline_v7"
export LD_LIBRARY_PATH=/home/nxcolelxu/unid/venv/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export PYTHONWARNINGS=ignore
nvidia-smi --query-gpu=name,memory.total --format=csv
/home/nxcolelxu/unid/venv/bin/python exp_retrieval.py --data_dir /home/nxcolelxu/unid/data/battery_data_light --device cuda
