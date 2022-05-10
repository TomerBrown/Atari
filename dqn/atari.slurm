#! /bin/sh
#SBATCH --job-name=atari
#SBATCH --output=atari.out
#SBATCH --error=atari.err
#SBATCH --partition=studentbatch
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python main.py