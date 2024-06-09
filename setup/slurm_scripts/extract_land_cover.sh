#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64000mb
#SBATCH --job-name=extract_land_cover
#SBATCH --mail-type=END
#SBATCH --output=extract_land_cover.log

eval "$(/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/setup/miniforge3/bin/conda shell.bash hook)"
conda activate thesis

python /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code/main/03_03_extract_data.py