# INPUT='./data/mturk/contexts/CSBF-contexts-2.csv'
# OUTPUT='./data/mturk/contexts'
BAR="$1"

if [ -z " $BAR " ] ; then
    echo "does not specify the number of annotator approve bar, defaults to 1"
    BAR=1
fi
INPUT='./data/mturk/advContexts/mturk_formal_1.csv'
OUTPUT='./data/mturk/advContexts/'

python tools/for_mturk/mturk_analysis_adv.py \
    --input_file $INPUT \
    --output_folder $OUTPUT \
    --binary \
    --record_annotation_summary \
    --task 'adv' \
    --number_of_annotators 3 \
    --bar $BAR \
    --boundary 0 \

    # Approve when the score is larger than the boundary
    # Approve when there are more annotators than the bar
