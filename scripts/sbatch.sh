#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=110000
#SBATCH --array=1-10

dataset="wikiLSHTC"
n=$SLURM_ARRAY_TASK_ID                  # define n
line=`sed "${n}q;d" ${dataset}_cmds.txt`    # get n:th line (1-indexed) of the file

# Do whatever with arrayparams.txt
srun $line
# e.g. srun ./my_program $line
