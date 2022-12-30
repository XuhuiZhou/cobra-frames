import os
import sys

import numpy as np
import pandas as pd

from tools.combine_toxigen_contexts import clean_for_mturk

np.random.seed(0)


def tag_explanations(df):
    """Tag the explanations with the corresponding model and prompt"""
    df["model"] = [
        "davinci-002" if i < 12612 else "davinci-003" for i in range(len(df))
    ]
    return df


def main():
    file = (
        "data/inference_data/toxigen_explanations_v2/toxigen_explanations.csv"
    )
    previous_mturk_file = (
        "data/inference_data/toxigen_explanations/toxigen_mturk_2.csv"
    )
    df = pd.read_csv(file)
    # # temporary info for the 002& 003 difference
    # df = tag_explanations(df)
    # df.to_csv("data/inference_data/toxigen_explanations/toxigen_explanations_tagged.csv", index=False)
    df_mturk_previous = pd.read_csv(previous_mturk_file)
    df = df[~df["id"].isin(df_mturk_previous["id"])]
    sampled_statements = np.random.choice(df["statement"].unique(), 35)
    df_mturk = df.groupby("statement").filter(
        lambda x: x["statement"].values[0] in sampled_statements
    )
    num = df_mturk["statement"].nunique()
    print(f"There are {num} unique statements in the sampled dataframe")
    # df_mturk = df[-range_start:-range_end]
    # df_mturk = df.sample(n=500, random_state=1)
    df_mturk.fillna("none", inplace=True)
    df_mturk = clean_for_mturk(df_mturk)
    df_mturk = df_mturk.drop(
        columns=["examples", "prompt", "statementCheck", "generation_method"]
    )
    print("the number of rows in the dataframe is {}".format(len(df_mturk)))
    df_mturk.to_csv(
        "data/inference_data/toxigen_explanations_v2/toxigen_mturk.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
