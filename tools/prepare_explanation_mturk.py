import os
import sys

import pandas as pd

from tools.combine_toxigen_contexts import clean_for_mturk


def main():
    file = "data/inference_data/toxigen_explanations/toxigen_explanations.csv"
    range_start = 220
    range_end = 120
    df = pd.read_csv(file)
    # df_mturk = df[-range_start:-range_end]
    df_mturk = df.sample(n=100, random_state=1)
    df_mturk.fillna("none", inplace=True)
    df_mturk = clean_for_mturk(df_mturk)
    df_mturk = df_mturk.drop(
        columns=["examples", "prompt", "statementCheck", "generation_method"]
    )
    # temporary info for the 002& 003 difference
    df_mturk["model"] = [
        "davinci-002" if i < 50 else "davinci-003"
        for i in range(len(df_mturk))
    ]
    print("the number of rows in the dataframe is {}".format(len(df_mturk)))
    df_mturk.to_csv(
        "data/inference_data/toxigen_explanations/toxigen_mturk_1.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
