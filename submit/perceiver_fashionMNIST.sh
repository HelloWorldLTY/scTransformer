#!/bin/bash
#SBATCH --job-name=perceiver_fashionMNIST
#SBATCH --output=./perceiver_fashionMNIST_gpu_job.txt
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx5000:1
#SBATCH --partition=gpu

module load CUDA
module load cuDNN
# using your anaconda environment
source ~/miniconda3/bin/activate

python3 /home/jl3773/scTransformer/main.py \
--fashionMNIST \
True \
--data_path \
/gpfs/ysm/scratch60/dijk/jl3773/perceiver_fashionMNIST \
--output_dir \
/gpfs/ysm/scratch60/dijk/jl3773/perceiver_fashionMNIST \
--fix_number_gene_crop \
False \
--batch_size_per_gpu \
16 \
--model_name \
Perceiver \
--depth \
8 \
--num_latents \
16 \
--num_workers \
4

