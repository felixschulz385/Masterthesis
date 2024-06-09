#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=08:00:00
#SBATCH --job-name=vscode

cd /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis

module load devel/code-server
code-server --bind-addr 0.0.0.0:8081 --auth password /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/setup/tu_zxobe27-master_thesis.code-workspace
