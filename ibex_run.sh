#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=3 
#SBATCH --gres=gpu:4 
#SBATCH --mem=40GB 
#SBATCH --time=1:59:58 
#SBATCH --partition=batch 
#SBATCH --begin=now+0hour

#SBATCH --job-name=anyres
#SBATCH --output=out/%x-%j-slurm.out

#SBATCH --signal=B:USR1@600
#SBATCH --reservation=A100

sig_handler()
{
echo "Force closing connection" && exit 0
}

# get tunneling info 
export XDG_RUNTIME_DIR="./tmp" node=$(hostname -s) 

cd anyres-gan
trap 'sig_handler' USR1 
# Run script 
source run_sh/run4m_l_teacher.sh

