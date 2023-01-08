# INPUT='./data/mturk/contexts/CSBF-contexts-2.csv'
# OUTPUT='./data/mturk/contexts'
INPUT='./data/mturk/advContexts/mturk_1.csv'
OUTPUT='./data/mturk/advContexts/'

python tools/for_mturk/mturk_analysis_adv.py \
    --input_file $INPUT \
    --output_folder $OUTPUT \
    --binary \
    --record_annotation_summary \
    --task 'adv' \
    --number_of_annotators 3 \
    --bar 1 \
    --boundary 0 \

    # Approve when the score is larger than the boundary
    # Approve when there are more annotators than the bar
