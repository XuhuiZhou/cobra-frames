#!/bin/sh

#INPUT='./data/mturk/CSBF-Verification-Pilot-1.csv'
#INPUT='./data/mturk/CSBF-Verification-Pilot-2-incomplete.csv'
#INPUT='./data/mturk/CSBF-Verification-Contexts-Pilots.csv'
#INPUT='./data/mturk/Context-Pilots-2-Annotation.csv'
INPUT='./data/mturk/contexts/CSBF-contexts-2.csv'
OUTPUT='./data/mturk/contexts'
# INPUT='./data/mturk/explanations_v2/mturk_1.csv'
# OUTPUT='./data/mturk/explanations_v2'
# INPUT='./data/mturk/t5_xxl/mturk_final.csv'
# OUTPUT='./data/mturk/t5_xxl'
# INPUT='./data/mturk/t5_xl_wo_context/mturk_final.csv'
# OUTPUT='./data/mturk/t5_xl_wo_context'
# INPUT='./data/mturk/t5_xl/mturk_final.csv'
# OUTPUT='./data/mturk/t5_xl'
# INPUT='./data/mturk/advContexts/mturk_formal_1.csv'
# OUTPUT='./data/mturk/advContexts/'

python tools/for_mturk/mturk_analysis.py \
    --input_file $INPUT \
    --output_folder $OUTPUT \
    --binary \
    --record_annotation_summary \
    --task 'context' \
    --number_of_annotators 2 \
    --bar 1\
    --boundary 0 \
    #--suggestion

    # Approve when the score is larger than the boundary
    # Approve when there are more annotators than the bar
