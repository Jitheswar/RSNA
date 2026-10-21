#!/usr/bin/env bash
#SBATCH --job-name=r6inf
#SBATCH --partition=cluster1-gpu,cluster1-gpu-add
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ch.jitheswar/RSNA/scratch_r6/r6-%x-%j.out
cd /home/ch.jitheswar/RSNA/scratch_r6
export PYTHONPATH=$HOME/r6-site SKIP_WHEELS=1 R6_TMP=/tmp/r6_${SHARD}_$SLURM_JOB_ID
nvidia-smi -L
~/rsna-gpu2/bin/python -u run_r6.py $SHARD
rm -rf /tmp/r6_${SHARD}_$SLURM_JOB_ID
