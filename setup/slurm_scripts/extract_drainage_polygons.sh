#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=90000mb
#SBATCH --job-name=extract_drainage_polygons
#SBATCH --mail-type=END
#SBATCH --output=extract_drainage_polygons.log

eval "$(/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/setup/miniforge3/bin/conda shell.bash hook)"
conda activate thesis

python /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code/main/02_02_extract_drainage_polygons.py