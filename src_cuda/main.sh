#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -t 0-00:59:00
#SBATCH -J main

#SBATCH -o main-%j.out -e main-%j.err

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module purge
module load cuda

nvcc main.cu md5c.cu -Xcompiler -Wall -Xcompiler -O3 -Xptxas -O3 -o main_o3  

./main_o3 8430894cfeb54a3625f18fe24fce272e