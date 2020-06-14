#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16000
#SBATCH --partition=gpu
#SBATCH --account=gpu
htop
watch -n 0.1 nvidia-smi
