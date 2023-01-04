# INPUT='./data/mturk/contexts/CSBF-contexts-2.csv'
# OUTPUT='./data/mturk/contexts'
INPUT='./data/mturk/explanations_v2_s/mturk_2.csv'
OUTPUT='./data/mturk/explanations_v2_s'

python tools/for_mturk/mturk_analysis_sen.py \
    --input_file $INPUT \
    --output_folder $OUTPUT \
    --binary \
    --record_annotation_summary \
    --task 'sensitivity' \
    --number_of_annotators 3 \
    --bar 1 \
    --boundary 0 \

    # Approve when the score is larger than the boundary
    # Approve when there are more annotators than the bar
