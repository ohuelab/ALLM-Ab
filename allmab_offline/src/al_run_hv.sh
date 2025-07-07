#!/bin/sh
batch_size=${2:-25}
python al_run_hv.py $1 --batch_size $batch_size

