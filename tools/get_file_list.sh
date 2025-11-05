#!/bin/bash

# ====== CONFIG ======


SEARCH_DIR="/standard/UVA-DSA/NIST EMS Project Data/EgoEMS_AAAI2026/"   # change this
SEARCH_DIR="/standard/UVA-DSA/NIST EMS Project Data/Review_Harvard_Dataverse"
# SEARCH_DIR="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/val_root/"   # change this
OUTPUT_FILE="./harvard_current_egoems_aaa2026_file_paths.txt"          # output text file name


# Recursively find all .mp4 files and save absolute paths
find "$SEARCH_DIR" -type f -name "*deidentified.mp4" -print | sort > "$OUTPUT_FILE"

# Optional: print summary
echo "Saved $(wc -l < "$OUTPUT_FILE") video paths to $OUTPUT_FILE"
