#!/bin/bash
#SBATCH --account=rrg-ashique
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --time=01-00:00
#SBATCH --mem-per-cpu=2048M
#SBATCH --array=1-30
#SBATCH --signal=USR1@90
#SBATCH --job-name=jsac			# single job name for the array
#SBATCH --gres=gpu:1


module load StdEnv/2023 gcc opencv cuda/12.2 python/3.10 mujoco/3.1.6
source ~/scratch/jaxrl2/.env/bin/activate

python /home/vasan/scratch/jaxrl2/examples/train_online.py --env_name HalfCheetah-v4