#!/bin/bash

#SBATCH -p htc                      # Use fn2 partition

#SBATCH --mem=1G                 

#SBATCH -t 0-1:00                   # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

module purge
module load anaconda2/4.4.0
source activate imitation
export PYTHONPATH=$PYTHONPATH:/home/mdrolet/imitation_baseline/imitation
python scripts/imitate_mj.py --mode bclone --env CartPole-v0 --data imitation_runs/classic/trajs/trajs_cartpole.h5 --limit_trajs 1 --data_subsamp_freq 10 --max_iter 1001 --bclone_eval_freq 100 --sim_batch_size 1 --log imitation_runs/classic/checkpoints/alg=bclone,task=cartpole,num_trajs=1,run=0.h5