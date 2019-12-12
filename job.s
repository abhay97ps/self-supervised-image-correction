#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=48:00:00
#SBATCH --mem=100000
#SBATCH --job-name=aps647
#SBATCH --mail-user=aps647@nyu.edu
#SBATCH --output=exp2.out


. ~/.bashrc
source activate aps647
python src/main.py
