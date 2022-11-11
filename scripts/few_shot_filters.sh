#!/bin/sh

INPUT="./data/cache/annotation_summary_all_approve.csv"
OUTPUT="./data/cache/annotation_summary_all_approve.t5.csv"

# INPUT="./data/cache/annotation_summary.csv"
# OUTPUT="./data/cache/annotation_summary.t5.csv"

python filters/few_shot_filter.py \
    --input_file $INPUT \
    --output_file $OUTPUT \