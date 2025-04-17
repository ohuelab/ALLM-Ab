#!/bin/sh

WDIR=$1
NSTRUCT=${2:-10}
OFFSET=${3:-50}
max_cpus=${4:-40}
INPUT=$WDIR/inputs
OUTPUT=$WDIR/outputs

cd /path/to/flex_ddG_tutorial

python -u run_flex_ddG_split.py \
    --input_path $INPUT \
    --output_path $OUTPUT \
    --max_cpus $max_cpus  --nstruct $NSTRUCT \
    --index 0 \
    --offset $OFFSET

