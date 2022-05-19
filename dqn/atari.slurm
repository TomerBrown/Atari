#! /bin/sh
#SBATCH --job-name=atari_default
#SBATCH --output=atari_default.out
#SBATCH --error=atari_default.err
#SBATCH --partition=studentbatch
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python main.py