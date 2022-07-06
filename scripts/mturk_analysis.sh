#!/bin/sh

#INPUT='./data/mturk/CSBF-Verification-Pilot-1.csv'
INPUT='./data/mturk/CSBF-Verification-Pilot-2-incomplete.csv'

OUTPUT='./data/mturk'

python tools/mturk_analysis.py \
    --input_file $INPUT \
    --output_folder $OUTPUT \
    --bar 1 \
#    --binary