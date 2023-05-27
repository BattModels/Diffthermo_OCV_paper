#!/bin/bash                                                                                      
#SBATCH -w d011                                                                                            
#SBATCH -J jaxthermo
#SBATCH -n 54 # Number of total cores
#SBATCH -N 1 # Number of nodes                                          
#SBATCH -p cpu
#SBATCH -A venkvis
#SBATCH --mem-per-cpu=2000 # Memory pool for all cores in MB (see also --mem-per-cpu)                        
#SBATCH -e error_%j.err
#SBATCH -o out_%j.out # File to which STDOUT will be written %j is the job # 
#SBATCH -t 1-00:00:00

#mkdir pb
source /home/mingzeya/miniconda3.sh
conda activate base
export JAX_ENABLE_X64=True
echo "Job started on `hostname` at `date`" 
python3 train.py

echo " "
echo "Job Ended at `date`"
