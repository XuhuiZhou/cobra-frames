# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/explain_model/train_explain_model.sh
# ./scripts/explain_model/train_explain_model.sh
MODEL_DIR='.log/explain-model-xl-w-o-context'

python sbf_modeling/inference.py \
    --gin_file="scripts/explain_model/explain_model_inference.gin" \
    --gin.MODEL_DIR="$MODEL_DIR" \
    --gin.MODE="'deployment'" \
    --gin.BATCH_SIZE=4
