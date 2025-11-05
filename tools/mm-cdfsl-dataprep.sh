#!/bin/bash
#SBATCH --job-name="MM-CDFSL-DataPrep"
#SBATCH --error="./logs/job-%j-mm-cdfsl-dataprep.err"
#SBATCH --output="./logs/job-%j-mm-cdfsl-dataprep.output"
#SBATCH --partition=standard
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account=uva-dsa

# ---- Config ----
video_list="./val_egoems_aaa2026_file_paths.txt"
output_root="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/MM-CDFSL_Format/Test/"
dataset_name="EgoEMS_AAAI2026"
frame_rate_stride=1
digits=10
overwrite=true   # true or false

# ---- Environment ----
module purge
module load miniforge
source /home/cjh9fw/.bashrc
echo "Running on: $HOSTNAME"
conda deactivate
conda activate egoexoems

# ---- Command ----
cmd=(python -u build_rgb_dataset.py
  --video-list "$video_list"
  --output-root "$output_root"
  --dataset-name "$dataset_name"
  --frame-rate-stride "$frame_rate_stride"
  --digits "$digits"
)

if [ "$overwrite" = true ]; then
  cmd+=(--overwrite)
fi

echo "Executing: ${cmd[*]}"
"${cmd[@]}"

echo "Done"
