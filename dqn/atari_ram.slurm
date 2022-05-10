#! /bin/sh
#SBATCH --job-name=atari_ram
#SBATCH --output=atari_ram.out
#SBATCH --error=atari_ram.err
#SBATCH --partition=studentbatch
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python ram.py