import pandas as pd

model = "t5_xl_wo_context"
datafile = "./data/inference_data/toxigen_explanations_v2/toxigen_explanations_val.csv"
predictionfile = f"./data/inference_data/{model}/answer.csv"

savedfile = f"./data/inference_data/{model}/toxigen_explanations_val.csv"

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
