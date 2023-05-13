#!/bin/sh

#EXAMPLE='./data/prompts/examples_explanations_expand.v2.csv'
EXAMPLE='./data/prompts/examples_explanations.v2.csv'

#### pilot two
# INPUT='./data/source_data/mAgr.onlyQuotes.csv'
# OUTPUT='./data/inference_data/mAgr.inference2_ex.csv'
# INPUT='./data/source_data/SBIC.v2.agg.trn.csv'
# OUTPUT='./data/inference_data/sbic.inference_ex.csv'

echo "Populate the 1 mAgr file with explanations"
# INPUT="./data/mturk/advContexts/advContexts_final.csv"
# SUB_FOLDER="explanations/"
# OUTPUT_FOLDER="./data/inference_data/adversarial_contexts_statements/explanations_v3_original_prompt/"

# INPUT="./data/mturk/advContexts/advContexts_exp.csv"
# SUB_FOLDER="explanations/"
# OUTPUT_FOLDER="./data/inference_data/adversarial_contexts_statements/explanations_v3_exp/"

# Second batch
INPUT="./data/mturk/advContexts/advContexts.csv"
SUB_FOLDER="explanations/"
OUTPUT_FOLDER="./data/inference_data/adversarial_contexts_statements/explanations_v3_original_prompt_secondBatch/"


python ./gpt3_generation/populateExplanationswithGPT3.py \
    --input_file $INPUT \
    --example_file $EXAMPLE \
    --sub_dir $SUB_FOLDER \
    --output_dir $OUTPUT_FOLDER \
    --output_file "mgr_contexts_explanations.csv"
