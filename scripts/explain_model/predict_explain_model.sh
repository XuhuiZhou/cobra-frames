# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/explain_model/predict_explain_model.sh
# ./scripts/explain_model/predict_explain_model.sh [experiment_name]
## Default evaluate metric is BLEU only

EXP="$1"
EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']"

if [ -z " $EXP " ] ; then
    echo "Please specify the experiment name"
    exit 1
fi

if [[ $EXP == "xxl" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xxl'" \
        --gin.EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']" \
        --gin.RESULT_FILE="'.log/explain-model-xxl/results.csv'" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_greedy" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/greedy.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl'" \
        --gin.RESULT_FILE="'.log/explain-model-xl/greedy-results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_topp" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/topp.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl'" \
        --gin.OUTPUT_DIR="'.log/explain-model-xl/topp'" \
        --gin.RESULT_FILE="'.log/explain-model-xl/topp-results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_topk" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/topk.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl'" \
        --gin.RESULT_FILE="'.log/explain-model-xl/topk-results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_w_o_context" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/without_context.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl-w-o-cotext'" \
        --gin.RESULT_FILE="'.log/explain-model-xl-w-o-cotext/results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl'" \
        --gin.RESULT_FILE="'.log/explain-model-xl/results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "large" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin.MODEL_DIR="'.log/explain-model-large'" \
        --gin.RESULT_FILE="'.log/explain-model-large/results.csv'" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "base" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin.MODEL_DIR="'.log/explain-model-base'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.RESULT_FILE="'.log/explain-model-base/results.csv'" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=16
elif [[ $EXP == "small" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin.MODEL_DIR="'.log/explain-model-small'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.RESULT_FILE="'.log/explain-model-small/results.csv'" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=16
elif [[ $EXP == "xl_adv" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl'" \
        --gin.OUTPUT_DIR="'.log/explain-model-xl/adv'" \
        --gin.RESULT_FILE="'.log/explain-model-xl/results_adv.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4

    python tools/sbf_modeling/evaluate_advContext.py \
        --prediction_file ".log/explain-model-xl/adv/answer.csv"

elif [[ $EXP == "xl_wo_context_adv" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin_file="exp/configs/without_context.gin" \
        --gin.MODEL_DIR="'.log/explain-model-xl-w-o-cotext'" \
        --gin.OUTPUT_DIR="'.log/explain-model-xl-w-o-cotext/adv'" \
        --gin.RESULT_FILE="'.log/explain-model-xl-w-o-cotext/results_adv.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=4

    python tools/sbf_modeling/evaluate_advContext.py \
        --prediction_file ".log/explain-model-xl-w-o-cotext/adv/answer.csv"

elif [[ $EXP == "small_adv" ]]; then
    python sbf_modeling/inference.py \
        --gin_file="scripts/explain_model/explain_model_inference.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin.MODEL_DIR="'.log/explain-model-small'" \
        --gin.OUTPUT_DIR="'.log/explain-model-small/adv'" \
        --gin.RESULT_FILE="'.log/explain-model-small/results_adv.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'deployment'" \
        --gin.BATCH_SIZE=16

    python tools/sbf_modeling/evaluate_advContext.py \
        --prediction_file ".log/explain-model-small/adv/answer.csv"
else
    echo "Experiment name not found"
    exit 1
fi
