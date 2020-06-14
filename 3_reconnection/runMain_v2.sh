#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --nodelist=biomix30

export PATH=${PATH}:/opt/MATLAB/R2018b/bin 

cd matlabCodes/ 
matlab -nodisplay -nosplash -nojvm -r "main('../input/', '../output/') "
