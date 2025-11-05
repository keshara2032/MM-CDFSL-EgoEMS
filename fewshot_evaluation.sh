#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="MM-CDFSL"
#SBATCH --error="./logs/job-%j-mm_cdfsl_fewshot_eval_script.err"
#SBATCH --output="./logs/job-%j-mm_cdfsl_fewshot_eval_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

CHKPOINT_PATH="/standard/UVA-DSA/Keshara/MM-CDFSL/checkpoints/ours_ego4d-epic.pt"
N_WAY=20
K_SHOT=5

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda deactivate &&
conda activate egoems &&
python -u lit_main_mmdistill.py train=False test=True ckpt_path="$CHKPOINT_PATH" data_module.n_way="$N_WAY" data_module.k_shot="$K_SHOT" &&
echo "Done" &&
exit
