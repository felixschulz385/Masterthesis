#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32000mb
#SBATCH --job-name=compute_reachability_graph
#SBATCH --mail-type=END
#SBATCH --output=compute_reachability_graph.log

eval "$(/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/setup/miniforge3/bin/conda shell.bash hook)"
conda activate thesis

python /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code/main/03_00_reachability_graph.py