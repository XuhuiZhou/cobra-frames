#!/bin/sh

EXAMPLE='./data/prompts/adv_context.csv'
EXAMPLE='./data/prompts/examples.v2.context_statementAdv.csv'

#### pilot two
# INPUT='./data/source_data/mAgr.onlyQuotes.csv'
# OUTPUT='./data/inference_data/mAgr.inference2_ex.csv'
# INPUT='./data/source_data/SBIC.v2.agg.trn.csv'
# OUTPUT='./data/inference_data/sbic.inference_ex.csv'

echo "Generate the adversarial contexts and related statements"
OUTPUT_FOLDER="./data/inference_data/adversarial_contexts_statements/"

python ./gpt3_generation/populateAdvContextswithGPT3.py \
    --example_file $EXAMPLE \
    --output_dir $OUTPUT_FOLDER \
    --post 'This is the placeholder' \
    --dataset_size 20 \
    --n_examples 5
