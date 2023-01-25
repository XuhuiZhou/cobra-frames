import os
import sys

import numpy as np
import pandas as pd

from tools.combine_toxigen_contexts import clean_for_mturk

np.random.seed(0)


def transform_advContexts(df):
    # shuffle the columns
    df = df.copy()
    df_normal = df.sample(frac=0.5, random_state=1)
    df_normal = df_normal.reset_index(drop=True)
    df_normal.rename(
        columns={
            "harmless_situationalContext": "situationalContext1",
            "harmless_speakerIdentity": "speaker1",
            "harmless_listenerIdentity": "listener1",
            "offensive_situationalContext": "situationalContext2",
            "offensive_speakerIdentity": "speaker2",
            "offensive_listenerIdentity": "listener2",
        },
        inplace=True,
    )
    df_normal["offensive_position"] = [2] * len(df_normal)
    df_reversed = df[~df["statement"].isin(df_normal["statement"])]
    df_reversed = df_reversed.reset_index(drop=True)
    df_reversed.rename(
        columns={
            "harmless_situationalContext": "situationalContext2",
            "harmless_speakerIdentity": "speaker2",
            "harmless_listenerIdentity": "listener2",
            "offensive_situationalContext": "situationalContext1",
            "offensive_speakerIdentity": "speaker1",
            "offensive_listenerIdentity": "listener1",
        },
        inplace=True,
    )
    df_reversed["offensive_position"] = [1] * len(df_reversed)
    df = pd.concat([df_normal, df_reversed])
    # shuffle the rows
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    return df


def main():
    ## Hyperparameters
    # file = "data/inference_data/adversarial_contexts_statements/mAgr_contexts_cleaned.csv"
    file = (
        "data/inference_data/adversarial_contexts_statements/adv_contexts.csv"
    )
    # file = "data/inference_data/adversarial_contexts_statements/selfGen_contexts.csv"
    # # temporary info for the 002& 003 difference
    # df = tag_explanations(df)
    # df.to_csv("data/inference_data/toxigen_explanations/toxigen_explanations_tagged.csv", index=False)
    # previous_mturk_file = (
    #     "data/inference_data/toxigen_explanations_v2/toxigen_mturk.csv"
    # )
    saved_mturk_file = "data/inference_data/adversarial_contexts_statements/mturk_formal_1.csv"
    previous_mturk_file = [
        "data/inference_data/adversarial_contexts_statements/mturk_1.csv",
        "data/inference_data/adversarial_contexts_statements/mturk_3.csv",
    ]
    sample_num = 500

    df = pd.read_csv(file)
    if previous_mturk_file != None:
        # read all previous mturk samples
        for file in previous_mturk_file:
            df_mturk_previous = pd.read_csv(file)
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
    df_mturk = transform_advContexts(df_mturk)
    df_mturk = df_mturk.drop(columns=["examples"])

    print("the number of rows in the dataframe is {}".format(len(df_mturk)))
    df_mturk.to_csv(
        saved_mturk_file,
        index=False,
    )


if __name__ == "__main__":
    main()
