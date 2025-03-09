#!/bin/bash
set -e  # Exit on error

if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments"
    echo "Usage: $0 <detection_pkl_path> <config_path>"
    echo "Example: $0 projects/work_dirs_griffin_35scenes_40m_0216/vehicle-side/tiny_track_r50_stream_bs8_24epoch_3cls_coEval_smallRange/results-02171028.pkl projects/configs_griffin_35scenes_40m_0216/vehicle-side/tiny_track_r50_stream_bs8_24epoch_3cls_coEval_smallRange.py"
    exit 1
fi

# Input detection results
DET_OUTPUT_PKL=$1
CONFIG_PATH=$2

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Extract model name and parent directory
DETNAME=$(echo $DET_OUTPUT_PKL | awk -F'/' '{print $(NF-1)}')
PARENT_DIR="${PROJECT_ROOT}/$(dirname "${DET_OUTPUT_PKL}")"

# Extract dataset name from path
DATASET_NAME=$(echo "${DET_OUTPUT_PKL}" | grep -o 'griffin_[^/]*\|spd_[^/]*' | head -n1)
if [[ -z "${DATASET_NAME}" ]]; then
    DATASET_NAME="spd"
fi

# Configure paths based on dataset
if [[ "${DATASET_NAME}" == griffin* ]]; then
    SPLIT_DATA_PATH="${PROJECT_ROOT}/data/split_datas/${DATASET_NAME}.json"
    EVAL_RESULT_NAME="${DATASET_NAME}_late_camera"
    DATAROOT="${PROJECT_ROOT}/datasets/${DATASET_NAME}/griffin-nuscenes"
else
    SPLIT_DATA_PATH="${PROJECT_ROOT}/data/split_datas/cooperative-split-data-spd.json"
    EVAL_RESULT_NAME="spd_late_camera"
    DATAROOT="${PROJECT_ROOT}/datasets/V2X-Seq-SPD-Example"
fi

# Setup output directories
CAT="Car"
SPLIT="val"

OUTPUT_PATH_DTC="${PARENT_DIR}/detection_results_to_kitti"
OUTPUT_PATH_DTC_SUB="${OUTPUT_PATH_DTC}/${DETNAME}_${CAT}_${SPLIT}"
OUTPUT_PATH_TRACK="${PARENT_DIR}/tracking_results_to_kitti"
OUTPUT_PATH_TRACK_PKL="${PARENT_DIR}/tracking_results_to_nusc_pkl/${DETNAME}_${CAT}_${SPLIT}_AB3DMOT.pkl"

# Add AB3DMOT to Python path
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/projects/ab3dmot_plugin"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/projects/ab3dmot_plugin/Xinshuo_PyToolbox"

# Step 1: Convert detection results to KITTI format
echo "Converting detection results to KITTI format..."
mkdir -p "${OUTPUT_PATH_DTC_SUB}"
python "${PROJECT_ROOT}/tools/result_converter/det_result_nusc2kitti.py" \
    --input-pkl-path "${DET_OUTPUT_PKL}" \
    --output-dir-path "${OUTPUT_PATH_DTC_SUB}" \
    --data-root "${DATAROOT}"

# Step 2: Run AB3DMOT tracking
echo "Running AB3DMOT tracking..."
mkdir -p "${OUTPUT_PATH_TRACK}"
python "${PROJECT_ROOT}/projects/ab3dmot_plugin/main_tracking.py" \
    --dataset KITTI \
    --split "${SPLIT}" \
    --det_name "${DETNAME}" \
    --cat "${CAT}" \
    --split-data-path "${SPLIT_DATA_PATH}" \
    --input-path "${OUTPUT_PATH_DTC}" \
    --output-path "${OUTPUT_PATH_TRACK}"

# Step 3: Convert tracking results to NuScenes pkl format and calculate metrics
echo "Converting tracking results to NuScenes pkl format..."
python "${PROJECT_ROOT}/tools/result_converter/track_result_kitti2nusc.py" \
    --config "${PROJECT_ROOT}/${CONFIG_PATH}" \
    --out "${OUTPUT_PATH_TRACK_PKL}" \
    --input-path "${OUTPUT_PATH_TRACK}/${DETNAME}_${SPLIT}_H1/data_0"

echo "Tracking completed successfully!"