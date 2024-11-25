#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p venkvis-largemem
#SBATCH -t 04:00:00
#SBATCH -J LixFeSiO4_Chebyshev
#SBATCH --mem-per-cpu=4000 # Memory pool for all cores in MB
#SBATCH -o out_%j.out
#SBATCH -e error_%j.err
##SBATCH --open-mode=append # Needed to append, instead of overwriting the log
##SBATCH --mail-user=amyao@umich.edu 
##SBATCH --mail-type=ALL


# Create job-specific tmp directory
export TMPDIR=$(mktemp --directory --tmpdir)
# Ensure Cleanup on exit
trap 'rm -rf -- "$TMPDIR"' EXIT

echo "Job started on `hostname` at `date`"
source /home/amyao/miniconda3.sh
conda activate chgnet
python3 main.py
echo " "
echo "Job Ended at `date`"


