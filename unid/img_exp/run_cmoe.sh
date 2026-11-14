#!/bin/bash
#SBATCH --time      06:00:00
#SBATCH -c          8
#SBATCH --mem       40G
#SBATCH --gpus      1
#SBATCH --constraint ada
# 사용: sbatch -J <name> -o logs/<name>_%j.out run_cmoe.sh <python 인자...>
cd /home/nxcolelxu/unid/img_exp
export LD_LIBRARY_PATH=/home/nxcolelxu/unid/venv/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
nvidia-smi --query-gpu=name --format=csv,noheader
/home/nxcolelxu/unid/venv/bin/python -W ignore "$@"
