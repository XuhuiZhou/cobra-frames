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
for i in {0..30}
do
    echo "Populate the $i toxigen file"
    INPUT="./data/cleaned_data/toxigen_shuffled/toxigen_${i}.csv"
    OUTPUT="./data/inference_data/toxigen_shuffled/toxigen_${i}.csv"
    SEED=$i

    python populateSBFwithGPT3.v5.py \
        --input_file $INPUT \
        --example_file_context $EXAMPLE \
        --output_file $OUTPUT \
        --random_seed $SEED 
done