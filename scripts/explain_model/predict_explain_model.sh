# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/explain_model/train_explain_model.sh
# ./scripts/explain_model/train_explain_model.sh
# MODEL_DIR='.log/explain-model-xl-w-o-context'
EXP="$1"

if [ -z " $EXP " ] ; then
    echo "Please specify the experiment name"
    exit 1
fi

if [[ $EXP == "xxl" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xxl'" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_greedy" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/greedy.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl'" \
        --gin.RESULT_FILE="'.log/explain-model-xl/greedy-results.csv'" \
        --gin.EVLUATE_METRICS="['bleu', 'bertscore']" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_topp" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/topp.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl'" \
        --gin.RESULT_FILE="'.log/explain-model-xl/topp-results.csv'" \
        --gin.EVLUATE_METRICS="['bleu', 'bertscore']" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_w_o_context" ]]; then
    python sbf_modeling/inference.py \
    --gin_file="scripts/explain_model/explain_model_inference.gin" \
    --gin_file="exp/configs/without_context.gin" \
    --gin.MODEL_DIR="'.log/explain-model-xl-w-o-cotext'" \
    --gin.MODE="'deployment'" \
    --gin.BATCH_SIZE=4
elif [[ $EXP == "xl" ]]; then
    python sbf_modeling/inference.py \
    --gin_file="scripts/explain_model/explain_model_inference.gin" \
    --gin.MODEL_DIR="'.log/explain-model-xl'" \
    --gin.MODE="'deployment'" \
    --gin.BATCH_SIZE=4
else
    echo "Experiment name not found"
    exit 1
fi
