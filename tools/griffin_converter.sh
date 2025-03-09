#!/bin/bash

# Check if an argument was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_prefix>"
    echo "Valid dataset_prefix values: griffin_50scenes_25m, griffin_50scenes_40m, griffin_100scenes_random"
    exit 1
fi

# Assign the first argument to dataset_prefix
dataset_prefix=$1

# Validate dataset_prefix
valid_prefixes=("griffin_50scenes_25m" "griffin_50scenes_40m" "griffin_100scenes_random")
valid=0
for prefix in "${valid_prefixes[@]}"; do
    if [ "$dataset_prefix" == "$prefix" ]; then
        valid=1
        break
    fi
done

if [ $valid -eq 0 ]; then
    echo "Error: Invalid dataset_prefix. Valid values are: griffin_50scenes_25m, griffin_50scenes_40m, griffin_100scenes_random"
    exit 1
fi

echo "Using dataset_prefix: $dataset_prefix"

# convert kitti format to nuscenes format
python tools/griffin_data_converter/trans_kitti2nuscenes.py \
    --source_dir datasets/${dataset_prefix}/griffin-release \
    --target_dir datasets/${dataset_prefix}/griffin-nuscenes \
    --split_file data/split_datas/${dataset_prefix}.json \
    --early_fusion

# generate data info pkl files
python tools/griffin_data_converter/generate_nuscenes_pkl.py \
    --root_path datasets/${dataset_prefix}/griffin-nuscenes \
    --out_path data/infos/${dataset_prefix} \
    --split_file data/split_datas/${dataset_prefix}.json \
    --early_fusion \
    --delay
