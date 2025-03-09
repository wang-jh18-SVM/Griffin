#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #                                            #    
# -------------------------------------------------- #
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    --out ${WORK_DIR}results-$T.pkl \
    2>&1 | tee ${WORK_DIR}logs/eval.$T