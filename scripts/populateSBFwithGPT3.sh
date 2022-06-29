#!/bin/sh

EXAMPLE='./data/examples.v2.toxonly.csv'
#### pilot one
#INPUT='./data/sbic.trn.r60.gpt3socCont.csv'
#OUTPUT='./data/sbic.inference.csv'
#INPUT='./data/mAgr.r60.gpt3socCont.csv'
#OUTPUT='./data/mAgr.inference.csv'

#### pilot two
#INPUT='./data/source_data/mAgr.onlyQuotes.csv'
#OUTPUT='./data/inference_data/mAgr.inference2.csv'
INPUT='./data/source_data/SBIC.v2.agg.trn.csv'
OUTPUT='./data/inference_data/sbic.inference2.csv'
SEED=42

python populateSBFwithGPT3.v2.py \
    --input_file $INPUT \
    --example_file $EXAMPLE \
    --output_file $OUTPUT \
    --random_seed $SEED \
    --sample 40 \