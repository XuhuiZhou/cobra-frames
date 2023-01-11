#!/bin/sh

EXAMPLE='./data/prompts/examples.v2.contextOnlyDiverse.csv'

for i in {501..502}
do
    echo "Populate the $i toxigen file"
    INPUT="./data/cleaned_data/toxigen_shuffled/toxigen_${i}.csv"
    OUTPUT="./data/inference_data/toxigen_shuffled/toxigen_${i}.csv"
    SEED=$i

    python ./gpt3_generation/populateContextswithGPT3.py \
        --input_file $INPUT \
        --example_file $EXAMPLE \
        --output_file $OUTPUT \
        --random_seed $SEED
done
echo "Done generating contexts for toxigen files"
echo "Combining all toxigen files"
python tools/combine_toxigen_contexts.py toxigen_6gen 401 500
echo "Done combining all toxigen files"