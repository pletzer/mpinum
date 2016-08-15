#!/bin/bash

#SBATCH -J laplacian3d_1
#SBATCH -A nesi99999
#SBATCH --time=00:10:00
#SBATCH --ntasks=64
#SBATCH --mem=1G

module load Python/3.4.1-goolf-1.5.14
time srun python exLaplacian3d.py 256 10


