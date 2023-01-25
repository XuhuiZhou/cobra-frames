import pandas as pd

folder = "data/mturk/t5_xl/"
data_file1 = "mturk_1.csv"
data_file2 = "mturk_2.csv"
saved_mturk_file = "mturk_final.csv"

df_1 = pd.read_csv(f"{folder}{data_file1}")
df_2 = pd.read_csv(f"{folder}{data_file2}")

# concat the two files
df = pd.concat([df_1, df_2])
df.to_csv(f"{folder}{saved_mturk_file}", index=False)
