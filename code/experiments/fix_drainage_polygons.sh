#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32000mb
#SBATCH --job-name=fix_drainage_polygons
#SBATCH --mail-type=END
#SBATCH --output=fix_drainage_polygons.log

eval "$(/home/tu/tu_tu/tu_zxobe27/miniforge3/bin/conda shell.bash hook)"
conda activate thesis

python /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code/experiments/fix_drainage_polygons.py