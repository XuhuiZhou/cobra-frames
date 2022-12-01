# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/training_explain_model/train_explain_model.sh
# ./scripts/training_explain_model/train_explain_model.sh

python sbf_modeling/train.py --gin_file="scripts/training_explain_model/explain_model.gin" --gin.MODEL_DIR="'explain-model'" --gin.MODE="'deployment'"
