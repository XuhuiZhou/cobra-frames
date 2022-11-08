#!/bin/sh

#INPUT='./data/mturk/CSBF-Verification-Pilot-1.csv'
#INPUT='./data/mturk/CSBF-Verification-Pilot-2-incomplete.csv'
#INPUT='./data/mturk/CSBF-Verification-Contexts-Pilots.csv'
#INPUT='./data/mturk/Context-Pilots-2-Annotation.csv'
INPUT='./data/mturk/contexts/CSBF-contexts-1.csv'

OUTPUT='./data/mturk/contexts'

python tools/mturk_analysis.py \
    --input_file $INPUT \
    --output_folder $OUTPUT \
    --binary \
    --record_annotation_summary \
    --task 'context' \
    --number_of_annotators 3 \
    --bar 1 \
    --boundary 1 \
    --original_file './data/inference_data/toxigen_shuffled/toxigen_1gen/toxigen_mturk.csv'
    
    # Approve when the score is larger than the boundary 
    # Approve when there are more annotators than the bar