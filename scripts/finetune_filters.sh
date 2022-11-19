#!/bin/sh

INPUT="./data/filters"
OUTPUT="models/filters"

# INPUT="./data/cache/annotation_summary.csv"
# OUTPUT="./data/cache/annotation_summary.t5.csv"

python filters/finetuned_filter.py \
    --data_dir $INPUT \
    --model_dir $OUTPUT \
