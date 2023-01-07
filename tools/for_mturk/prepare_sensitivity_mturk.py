import os
import sys

import numpy as np
import pandas as pd

from tools.combine_toxigen_contexts import clean_for_mturk

np.random.seed(0)


def main():
    ## Hyperparameters
    file = (
        "data/inference_data/sensitivity/toxigen_explanations_valmaxdiff.csv"
    )
    # # temporary info for the 002& 003 difference
    # df = tag_explanations(df)
    # df.to_csv("data/inference_data/toxigen_explanations/toxigen_explanations_tagged.csv", index=False)
    # previous_mturk_file = (
    #     "data/inference_data/toxigen_explanations_v2/toxigen_mturk.csv"
    # )
    saved_mturk_file = "data/inference_data/sensitivity/mturk_1.csv"
    previous_mturk_file = None
    sample_num = 75

    df = pd.read_csv(file)
    if previous_mturk_file != None:
        df_mturk_previous = pd.read_csv(previous_mturk_file)
        df = df[~df["statement"].isin(df_mturk_previous["statement"])]

    if sample_num:
        df_mturk = df.sample(n=sample_num, random_state=1, replace=False)
    else:
        df_mturk = df

    num = df_mturk["statement"].nunique()
    print(f"There are {num} unique statements in the sampled dataframe")
    # df_mturk = df[-range_start:-range_end]
    # df_mturk = df.sample(n=500, random_state=1)
    df_mturk.fillna("none", inplace=True)
    df_mturk = clean_for_mturk(df_mturk)

    print("the number of rows in the dataframe is {}".format(len(df_mturk)))
    df_mturk.to_csv(
        saved_mturk_file,
        index=False,
    )


if __name__ == "__main__":
    main()
