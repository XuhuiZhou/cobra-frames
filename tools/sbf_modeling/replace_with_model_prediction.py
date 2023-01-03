import pandas as pd

datafile = "./data/inference_data/toxigen_explanations_v2/toxigen_explanations_val.csv"
predictionfile = "./data/inference_data/t5_xl/answer.csv"

savedfile = "./data/inference_data/t5_xl/val_with_model_predictions.csv"

df = pd.read_csv(datafile)
df_prediction = pd.read_csv(predictionfile)

relevant_col = [
    "intent",
    "targetGroup",
    "relevantPowerDynamics",
    "implication",
    "targetGroupEmotionalReaction",
    "targetGroupCognitiveReaction",
    "offensiveness",
]

df[relevant_col] = df_prediction[relevant_col]

df.to_csv(savedfile, index=False)
