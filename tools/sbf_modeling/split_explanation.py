import pandas as pd

folder = "data/inference_data/toxigen_explanations"

df = pd.read_csv(f"{folder}/toxigen_explanations.csv")
df = df.rename(columns={"speechContext": "situationalContext"})

df_train = df[:-1000]
df_val = df[-1000:]

df_train.to_csv(f"{folder}/toxigen_explanations_train.csv", index=False)
df_val.to_csv(f"{folder}/toxigen_explanations_val.csv", index=False)
