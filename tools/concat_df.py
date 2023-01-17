import pandas as pd

folder = "data/inference_data/adversarial_contexts_statements/"
data_file1 = "explanations_v3/mAgr_contexts_explanations.csv"
data_file2 = "explanations_v3_exp/exp_contexts_explanations.csv"
saved_mturk_file = "adv_contexts_explanations.csv"

df_1 = pd.read_csv(f"{folder}{data_file1}")
df_2 = pd.read_csv(f"{folder}{data_file2}")

# concat the two files
df = pd.concat([df_1, df_2])
df.to_csv(f"{folder}{saved_mturk_file}", index=False)
