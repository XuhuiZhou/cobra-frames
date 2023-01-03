#!/bin/sh

EXAMPLE='./data/prompts/examples_explanations.v2.csv'

#### pilot two
# INPUT='./data/source_data/mAgr.onlyQuotes.csv'
# OUTPUT='./data/inference_data/mAgr.inference2_ex.csv'
# INPUT='./data/source_data/SBIC.v2.agg.trn.csv'
# OUTPUT='./data/inference_data/sbic.inference_ex.csv'

echo "Populate the ${i} toxigen file with explanations"
INPUT="./data/inference_data/toxigen_shuffled/toxigen_${i}gen/toxigen_complete.csv"
SUB_FOLDER="toxigen_${i}gen"
OUTPUT_FOLDER="./data/inference_data/toxigen_explanations_v3/"

python ./gpt3_generation/populateExplanationswithGPT3.py \
    --input_file $INPUT \
    --example_file $EXAMPLE \
    --sub_dir $SUB_FOLDER \
    --output_dir $OUTPUT_FOLDER
