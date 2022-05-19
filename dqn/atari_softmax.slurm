#! /bin/sh
#SBATCH --job-name=atari_softmax
#SBATCH --output=atari_softmax.out
#SBATCH --error=atari_softmax.err
#SBATCH --partition=studentbatch
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python main.py --run_name softmax --action_selection softmax