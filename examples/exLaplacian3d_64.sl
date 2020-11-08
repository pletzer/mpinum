#!/bin/bash

#SBATCH -J laplacian3d_64
#SBATCH -A nesi99999
#SBATCH --time=00:10:00
#SBATCH --ntasks=64
#SBATCH --mem=1G

time srun python exLaplacian3d.py 256 10


