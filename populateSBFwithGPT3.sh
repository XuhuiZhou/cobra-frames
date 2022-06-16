#!/bin/sh
#INPUT='./data/sbic.trn.r60.gpt3socCont.csv'
#OUTPUT='./data/sbic.inference.csv'
INPUT='./data/mAgr.r60.gpt3socCont.csv'
OUTPUT='./data/mAgr.inference.csv'
EXAMPLE='./data/examples.v2.toxonly.csv'

python populateSBFwithGPT3.v2.py \
    --input_file $INPUT \
    --example_file $EXAMPLE
    --output_file $OUTPUT \