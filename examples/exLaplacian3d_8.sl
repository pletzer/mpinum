#!/bin/bash

#SBATCH -J laplacian3d_8
#SBATCH -A nesi99999
#SBATCH --time=00:10:00
#SBATCH --ntasks=8
#SBATCH --mem=8G

time srun python exLaplacian3d.py 512 10


