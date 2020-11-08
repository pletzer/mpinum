#!/bin/bash

#SBATCH -J laplacian3d_1
#SBATCH -A nesi99999
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=42G

time srun python exLaplacian3d.py 1000 20


