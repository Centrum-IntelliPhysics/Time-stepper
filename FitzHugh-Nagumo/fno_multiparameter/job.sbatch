#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -p a100
#SBATCH --account=sgoswam4_gpu
#SBATCH --gres=gpu:1
#SBATCH -n 12
#SBATCH -N 1
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

module load anaconda
conda activate my_torch
python3 main.py