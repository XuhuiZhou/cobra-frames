# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/explain_model/train_explain_model.sh
# ./scripts/explain_model/train_explain_model.sh


EXP="$1"

if [ -z " $EXP " ] ; then
    echo "Please specify the experiment name"
    exit 1
fi

if [[ $EXP == "train_xxl" ]]; then
    python sbf_modeling/train.py --gin_file="scripts/explain_model/explain_model.gin" --gin_file="exp/configs/train_explain_model_xxl_2_epoch.gin" --gin.MODEL_DIR="'.log/explain-model-xxl'" --gin.MODE="'deployment'"
elif [[ $EXP == "train_xl_w_o_context" ]]; then
    python sbf_modeling/train.py --gin_file="scripts/explain_model/explain_model.gin" --gin_file="exp/configs/train_explain_model_xl_2_epoch.gin" --gin_file="exp/configs/without_context.gin" --gin.MODEL_DIR="'.log/explain-model-xl-w-o-cotext'" --gin.MODE="'deployment'"
elif [[ $EXP == "train_xl" ]]; then
    python sbf_modeling/train.py --gin_file="scripts/explain_model/explain_model.gin" --gin_file="exp/configs/train_explain_model_xl_2_epoch.gin" --gin.MODEL_DIR="'.log/explain-model-xl'" --gin.MODE="'deployment'"
elif [[ $EXP == "train_large" ]]; then
    python sbf_modeling/train.py --gin_file="scripts/explain_model/explain_model.gin" --gin_file="exp/configs/train_explain_model_large_2_epoch.gin" --gin.MODEL_DIR="'.log/explain-model-large'" --gin.MODE="'deployment'"
else
    echo "Experiment name not found"
    exit 1
fi
