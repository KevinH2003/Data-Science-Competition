#!/usr/bin/env bash
#SBATCH -e slurm_log/err
#SBATCH --job-name=testing_job_lasso_xgb # Job name
#SBATCH --output=logs/testing_job__lasso_xgb_log_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=10gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:2
source ./
python3 -u comprehensive_testing.py