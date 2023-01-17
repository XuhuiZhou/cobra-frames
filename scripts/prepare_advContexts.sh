# /bin/bash
# This script should be excuted in the root directory of the project

# bash scripts/mturk_analysis/mturk_analysis_adv.sh 1
# python tools/sbf_modeling/transform_advContext.py \
#     --input_file './data/mturk/advContexts/annotation_summary.csv' \
#     --output_file './data/mturk/advContexts/advContexts.csv' \
#     --mode 'majority'

# bash scripts/mturk_analysis/mturk_analysis_adv.sh 2
python tools/sbf_modeling/transform_advContext.py \
    --input_file './data/mturk/advContexts/annotation_ex.csv' \
    --output_file './data/mturk/advContexts/advContexts_exp.csv' \
    --pre_output_file './data/mturk/advContexts/advContexts_ex.csv' \
    --mode 'allapp'
