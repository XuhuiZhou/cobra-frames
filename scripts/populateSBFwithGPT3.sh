#!/bin/sh

EXAMPLE='./data/examples.v2.contextOnlyDiverse.csv'
#EXAMPLE_CONT='./data/examples.v2.contextOnlyCombined.csv'
#### pilot one
#INPUT='./data/sbic.trn.r60.gpt3socCont.csv'
#OUTPUT='./data/sbic.inference.csv'
#INPUT='./data/mAgr.r60.gpt3socCont.csv'
#OUTPUT='./data/mAgr.inference.csv'

#### pilot two
# INPUT='./data/source_data/mAgr.onlyQuotes.csv'
# OUTPUT='./data/inference_data/mAgr.inference2_ex.csv'
# INPUT='./data/source_data/SBIC.v2.agg.trn.csv'
# OUTPUT='./data/inference_data/sbic.inference_ex.csv'
INPUT='./data/cleaned_data/toxigen.csv'
OUTPUT='./data/inference_data/toxigen.inference_ex.csv'
# INPUT='./data/source_data/example_candidate_toxigen.csv'
# OUTPUT='./data/inference_data/example_candidate.inference.csv'
SEED=1

python populateSBFwithGPT3.v5.py \
    --input_file $INPUT \
    --example_file $EXAMPLE \
    --output_file $OUTPUT \
    --random_seed $SEED \
    --sample 20 \
