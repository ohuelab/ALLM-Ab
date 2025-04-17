#!/bin/sh

batch_size=${2:-25}

python al_run_logits.py $1 --batch_size $batch_size
