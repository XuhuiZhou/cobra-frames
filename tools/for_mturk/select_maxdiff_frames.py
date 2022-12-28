import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from bert_score import score
from tqdm import tqdm

from tools.combine_toxigen_contexts import clean_for_mturk


def score_rule(i, j, v, method="bert_score"):
    """
    The rule to calculate the score.
    The score can come from one reference or multiple references.
    """
    if method == "bert_score":
        P, R, F1 = score([i], [j], lang="en", rescale_with_baseline=True)
    return F1[0].item()


def produce_pair_signals(contents: list) -> list:
    """
    Produce different pairs from the contents.
    """
    scores = []
    if len(contents) == 1:
        return [0]
    elif len(contents) == 2:
        return [1, 1]
    for i, j in combinations(contents, 2):
        scores.append(score_rule(i, j, None))
    # produce signals based on max score
    # TODO: make the signal process universal
    signals = [[1, 1, 0], [1, 0, 1], [0, 1, 1]][np.argmin(scores)]
    return signals


def get_pairs(df):
    """
    Get the pairs of the contexts and/or explanations.
    """
    context_var = [
        "situationalContext",
        "speakerIdentity",
        "listenerIdentity",
        "statement",
    ]
    explanation_var = [
        "intent",
        "targetGroup",
        "relevantPowerDynamics",
        "implication",
        "targetGroupEmotionalReaction",
        "targetGroupCognitiveReaction",
        "offensiveness",
    ]
    var = context_var + explanation_var
    pairs = df[var].stack().groupby(level=0).apply(",".join).to_list()
    return pairs


def prepare_mturk_data(df, num=200):
    """
    Prepare the mturk data.
    """
    sampled_statements = np.random.choice(df["statement"].unique(), num)
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
    return df_mturk


def main():
    anno_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "./data/inference_data/toxigen_explanations/toxigen_explanations_val.csv"
    )
    df = pd.read_csv(anno_file)
    df = prepare_mturk_data(df, num=100)
    pair_list = df.groupby("statement", sort=False).apply(get_pairs)
    overall_signals = []
    for i in tqdm(pair_list):
        print(produce_pair_signals(i))
        overall_signals += produce_pair_signals(i)
    df["signals"] = overall_signals
    df = df[df["signals"] == 1]
    df = df.drop(columns="signals")
    df.to_csv(
        "./data/mturk/explanations/toxigen_explanations_valmaxdiff.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
