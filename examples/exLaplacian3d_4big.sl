#!/bin/bash

#SBATCH -J laplacian3d_4
#SBATCH -A nesi99999
#SBATCH --time=00:30:00
#SBATCH --ntasks=4
#SBATCH --mem=40G

time srun python exLaplacian3d.py 1000 20


