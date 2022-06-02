#!/bin/sh
INPUT='./data/sbic.trn.r60.gpt3socCont.csv'
OUTPUT='./data/sbic.inference.csv'

python populateSBFwithGPT3.v2.py \
    --input_file $INPUT \
    --output_file $OUTPUT \