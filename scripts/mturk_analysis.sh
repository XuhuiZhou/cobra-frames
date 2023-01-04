#!/bin/sh

#INPUT='./data/mturk/CSBF-Verification-Pilot-1.csv'
#INPUT='./data/mturk/CSBF-Verification-Pilot-2-incomplete.csv'
#INPUT='./data/mturk/CSBF-Verification-Contexts-Pilots.csv'
#INPUT='./data/mturk/Context-Pilots-2-Annotation.csv'
# INPUT='./data/mturk/contexts/CSBF-contexts-2.csv'
# OUTPUT='./data/mturk/contexts'
# INPUT='./data/mturk/explanations_v2/mturk_1.csv'
# OUTPUT='./data/mturk/explanations_v2'
# INPUT='./data/mturk/t5_xl/mturk_1.csv'
# OUTPUT='./data/mturk/t5_xl'
INPUT='./data/mturk/t5_xl_wo_context/mturk_1.csv'
OUTPUT='./data/mturk/t5_xl_wo_context'

python tools/for_mturk/mturk_analysis.py \
    --input_file $INPUT \
    --output_folder $OUTPUT \
    --binary \
    --record_annotation_summary \
    --task 'explanation' \
    --number_of_annotators 3 \
    --bar 1\
    --boundary 1 \
    --suggestion

    # Approve when the score is larger than the boundary
    # Approve when there are more annotators than the bar
