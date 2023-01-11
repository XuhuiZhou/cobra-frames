# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/explain_model/train_and_predict.sh
# ./scripts/explain_model/train_and_predict.sh model_name

EXP="$1"

if [ "$EXP" != "large" ] && [ "$EXP" != "xl" ] && [ "$EXP" != "xxl" ] && [ "$EXP" != "xl_w_o_context" ] && [ "$EXP" != "small" ]; then
    echo "Experiment name not found"
    exit 1
fi

bash scripts/explain_model/train_explain_model.sh train_$EXP
bash scripts/explain_model/predict_explain_model.sh $EXP
