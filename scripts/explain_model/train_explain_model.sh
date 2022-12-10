# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/explain_model/train_explain_model.sh
# ./scripts/explain_model/train_explain_model.sh

python sbf_modeling/train.py --gin_file="scripts/explain_model/explain_model.gin" --gin_file="exp/configs/train_explain_model_large_batch_extensive.gin" --gin.MODEL_DIR="'.log/explain-model'" --gin.MODE="'deployment'"
