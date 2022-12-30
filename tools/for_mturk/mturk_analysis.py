import argparse
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from itertools import dropwhile

import numpy as np
import pandas as pd
import simpledorff
from agreement import computeAlpha, fleiss_kappa

# from disagree import metrics

Task_cols = {
    "explanation": (
        [
            "Answer.targetGroupRating",
            "Answer.intentRating",
            "Answer.implicationRating",
            "Answer.offensivenessRating",
            "Answer.powerDiffRating",
            "Answer.targetGroupEmoReactRating",
            "Answer.targetGroupCogReactRating",
        ],
        [
            "Answer.targetGroupSuggestion",
            "Answer.intentSuggestion",
            "Answer.implicationSuggestion",
            "Answer.offensivenessSuggestion",
            "Answer.powerDiffSuggestion",
            "Answer.targetGroupEmoReactSuggestion",
            "Answer.targetGroupCogReactSuggestion",
        ],
    ),
    "context": (
        [
            "Answer.listenerIdenRating",
            # 'Answer.situationRating',
            "Answer.psituationRating",
            "Answer.hsituationRating",
            "Answer.speakerIdenRating",
        ],
        [
            "Answer.listenerIdenSuggestion",
            # 'Answer.situationSuggestion',
            "Answer.psituationSuggestion",
            "Answer.hsituationSuggestion",
            "Answer.speakerIdenSuggestion",
        ],
    ),
}


def computeFleissKappa(df, col, groupCol, args):
    # Only works for binary setting.
    df = df.copy()
    if not args.binary:
        df[col] = (df[col] > args.boundary).astype(int)
    df = df[[groupCol, col]]
    df = df.groupby(by=[groupCol]).sum()
    # warning: hard-coded value here
    df["Rating_neg"] = args.number_of_annotators - df[col]
    score = fleiss_kappa(df.to_numpy(), method="randolf")
    return score


def record_annotation_summary(df_info, df, args):
    # TODO: fix this chunk of code; too hard to read
    warnings.warn(
        "This function only applies to three annotators scanario bc of hard-coded values."
    )
    # Read the original file before mturk annotation
    # df_ori = pd.read_csv(args.original_file)
    if args.suggestion:
        df_suggestion = df[[i for i in df.keys() if "Suggestion" in i]]
        # Warning: hard-coded value here
        df_result = list()
        df_suggestion_1 = df_suggestion[0::3]
        df_suggestion_2 = df_suggestion[1::3]
        df_suggestion_3 = df_suggestion[2::3]
        for j in range(0, 3 * len(df_suggestion_1), 3):
            row_dict = dict()
            for i in df_suggestion.keys():
                row_dict[i] = (
                    df_suggestion_1.loc[j, i]
                    + df_suggestion_2.loc[j + 1, i]
                    + df_suggestion_3.loc[j + 2, i]
                )
            df_result.append(row_dict)
        df_suggestion = pd.DataFrame(df_result)
    df = df[0::3]
    if args.task == "context":
        df = df[
            [
                "HITId",
                "Input.group",
                "Input.statement",
                "Input.speechContext",
                "Input.speakerIdentity",
                "Input.listenerIdentity",
            ]
        ]
        if args.suggestion:
            df = pd.concat(
                [
                    df.reset_index(drop=True),
                    df_suggestion.reset_index(drop=True),
                ],
                axis=1,
            )
        else:
            df = df.reset_index(drop=True)
        df_info = df.merge(df_info, on="HITId")

        relevant_cols = [
            "HITId",
            "Input.group",
            "Input.statement",
            "Input.speechContext",
            "Input.speakerIdentity",
            "Input.listenerIdentity",
            "Answer.hsituationRating",
            "Answer.psituationRating",
            "Answer.speakerIdenRating",
            "Answer.listenerIdenRating",
            "Answer.finalRating",
        ]

        if args.suggestion:
            relevant_cols += [
                "Answer.hsituationSuggestion",
                "Answer.psituationSuggestion",
                "Answer.speakerIdenSuggestion",
                "Answer.listenerIdenSuggestion",
            ]

        df_info = df_info[relevant_cols]
    else:
        df = df[
            [
                "HITId",
                "Input.group",
                "Input.id",
                "Input.statement",
                "Input.speechContext",
                "Input.speakerIdentity",
                "Input.listenerIdentity",
                "Input.targetGroup",
                "Input.intent",
                "Input.implication",
                "Input.offensiveness",
                "Input.relevantPowerDynamics",
                "Input.targetGroupEmotionalReaction",
                "Input.targetGroupCognitiveReaction",
            ]
        ]
        if args.suggestion:
            df = pd.concat(
                [
                    df.reset_index(drop=True),
                    df_suggestion.reset_index(drop=True),
                ],
                axis=1,
            )
        else:
            df = df.reset_index(drop=True)
        df_info = df.merge(df_info, on="HITId")

        relevant_cols = [
            "HITId",
            "Input.id_x",  # TODO: fix this
            "Input.group",
            "Input.statement",
            "Input.speechContext",
            "Input.speakerIdentity",
            "Input.listenerIdentity",
            "Input.targetGroup",
            "Input.intent",
            "Input.implication",
            "Input.offensiveness",
            "Input.relevantPowerDynamics",
            "Input.targetGroupEmotionalReaction",
            "Input.targetGroupCognitiveReaction",
            "Answer.targetGroupRating",
            "Answer.intentRating",
            "Answer.implicationRating",
            "Answer.offensivenessRating",
            "Answer.powerDiffRating",
            "Answer.targetGroupEmoReactRating",
            "Answer.targetGroupCogReactRating",
        ]

        if args.suggestion:
            relevant_cols += [
                "Answer.targetGroupSuggestion",
                "Answer.intentSuggestion",
                "Answer.implicationSuggestion",
                "Answer.offensivenessSuggestion",
                "Answer.powerDiffSuggestion",
                "Answer.targetGroupEmoReactSuggestion",
                "Answer.targetGroupCogReactSuggestion",
            ]

        df_info = df_info[relevant_cols]

    # binarize the rating
    for i in df_info.keys():
        if "Rating" in i:
            df_info[i] = df_info[i].astype(int)
    df_info.to_csv(f"{args.output_folder}/annotation_summary.csv", index=False)


def render_offensiveness_labels(offensiveness):
    offensive = []
    for i, row in enumerate(offensiveness.str.split(",", expand=False)):
        # if not isinstance(row, float):
        offensive.append(row[0])

    offensive = list(map(lambda x: x.lstrip().lower(), offensive))
    offensive = list(map(lambda x: x.replace("very ", ""), offensive))
    offensive = list(map(lambda x: x.replace("highly ", ""), offensive))
    offensive = list(map(lambda x: x.replace("extremely ", ""), offensive))

    for i, off in enumerate(offensive):
        if "anti-sem" in off:
            offensive[i] = "anti-semitic"
        if (
            "none" in off
            or "not offensive" in off
            or "not necessarily offensive" in off
            or "n/a" in off
            or "not applicable" in off
        ):
            offensive[i] = "not offensive"
        if "ableis" in off:
            offensive[i] = "ableist"
        if "racis" in off:
            offensive[i] = "racist"
        if "stereotype" in off:
            offensive[i] = "offensive stereotype"
    return offensive


def output_quality_ratio(df, col, args):
    df_info = df.copy()
    if not args.binary:
        # Special treatment for the offensiveness rating
        df["Answer.offensivenessRating"] = [
            args.boundary + 1 if j == "not offensive" and i == -1 else i
            for i, j in zip(
                df["Answer.offensivenessRating"],
                render_offensiveness_labels(df["Input.offensiveness"]),
            )
        ]
        df_info[col] = (df_info[col] > args.boundary).astype(int)
    df_info = df_info.groupby(by=["HITId"]).sum()
    df_info = df_info > args.bar
    # produce the final label
    if args.task == "context":
        df_info["Answer.finalRating"] = (
            df_info["Answer.hsituationRating"]
            & df_info["Answer.speakerIdenRating"]
            & df_info["Answer.listenerIdenRating"]
        ).astype(int)
    if args.record_annotation_summary:
        record_annotation_summary(df_info, df, args)
    sum_ = df_info.sum()
    total = len(df_info)
    frames = {}
    for i in col:
        frames[i] = [f"{100*sum_[i]/total:.2f}%"]
    df_info = pd.DataFrame.from_dict(frames)
    df_info = df_info.rename(index={0: "approval rate"})
    return df_info


def stats_of_interest(df, has_text=False):
    """
    Customized function to calculate the distribution stats of the table
    """
    if has_text:
        frames = {}
        l = len(df)
        for i in df.keys():
            num_has_text = l - df[i].str.contains("{}|none").sum()
            frames[i] = [num_has_text]
        stats = pd.DataFrame.from_dict(frames)
        return stats
    else:
        frames = []
        for i in df.keys():
            frames.append(df[i].value_counts())
        stats = pd.concat(frames, join="outer", axis=1)
        stats = stats.rename(
            index={
                3: "very likely",
                1: "somewhat unlikely",
                2: "somewhat likely",
                0: "very unlikely",
                -1: "none",
            }
        )
        return stats


def pretty(df):
    """
    Customized function to make the table better readable
    """
    keys = df.keys()
    new_keys = [i.split(".")[-1] for i in keys]
    df = df.rename(columns={i: j for i, j in zip(keys, new_keys)})
    return df


def select_workers(df, col):
    """Select workers that at least have raise one unlikely ratings."""
    workers = []
    for i in col:
        workers += df[df[i] < 2]["WorkerId"].tolist()
    workers = set(workers)
    return workers


def analyze_perHitTime(df, unix=True):
    if unix:
        df["WorkTimeInSeconds"] = (
            df["Answer.clickedSubmitTime"] - df["Answer.clickedConsentTime"]
        ) / 1000
        line = (df["WorkTimeInSeconds"] / 60).describe()
        line_ignoreMax = (
            df.sort_values("WorkTimeInSeconds")
            .groupby("WorkerId")
            .apply(lambda x: x[:-1])
        )
        line_ignoreMax = (line_ignoreMax["WorkTimeInSeconds"] / 60).describe()
    else:
        line = (df["WorkTimeInSeconds"] / 60).describe()
        line_ignoreMax = (
            df.sort_values("WorkTimeInSeconds")
            .groupby("WorkerId")
            .apply(lambda x: x[:-1])
        )
        line_ignoreMax = (line_ignoreMax["WorkTimeInSeconds"] / 60).describe()
    items = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    time_dict = defaultdict(list)
    for i in items:
        time_dict[i].append(line[i])
        time_dict[i].append(line_ignoreMax[i])
    df_time = pd.DataFrame.from_dict(time_dict)
    # df_time = df_time.rename('')
    return df_time


def calculate_agreement(df, col, args):
    """
    Utilize external function to calculate the agreement.
    """
    frames = {}
    for i in col:
        new_df = df[["HITId", "WorkerId", i]]
        new_df = new_df.rename(columns={i: "Rating"})
        scores = computeAlpha(new_df, "Rating", groupCol="HITId")
        fleiss_kappa_score = computeFleissKappa(
            new_df, "Rating", groupCol="HITId", args=args
        )
        # k.to_csv(f'pairwise_matrix_{i}.csv')
        # print(k)
        frames[i] = [
            f"{scores['ppa']*100:.2f}%",
            f"{scores['rnd_ppa']*100:.2f}%",
            f"{scores['alpha']:.4f}",
            f"{fleiss_kappa_score:.4f}",
        ]
    df = pd.DataFrame.from_dict(frames)
    df = df.rename(
        index={
            0: "pairwise agreement",
            1: "random agreement",
            2: "Krippendorf's alpha",
            3: "Fleiss' kappa",
        }
    )
    return df


def normalize_df(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


def main():
    """
    Script for analyzing Mturk produced data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_folder")
    parser.add_argument("--task", default="explanation", type=str)
    parser.add_argument("--suggestion", action="store_true")
    parser.add_argument(
        "--bar",
        default=1,
        type=int,
        help="Approve when there are more annotators than the bar",
    )
    parser.add_argument(
        "--boundary",
        default=1,
        type=int,
        help="Approve when the score is larger than the boundary",
    )
    parser.add_argument(
        "--number_of_annotators",
        default=1,
        type=int,
        help="Number of annotators",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="make the value of ratings to binary when calculating the agreement",
    )
    parser.add_argument(
        "--record_annotation_summary",
        action="store_true",
        help="record the annotation summary",
    )

    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    # #### TEMPORARY FEATURE: seperate two models' stats
    # df = df[df['Input.model'] == 'davinci-003']

    # df = df[df['source_tag']=='sbic']
    df = df.rename(
        columns={
            "Answer.targetGroupEffectSuggestion": "Answer.targetGroupCogReactSuggestion",
            "Answer.targetGroupReactionSuggestion": "Answer.targetGroupEmoReactSuggestion",
        }
    )
    # df_stats = stats_of_interest(df)
    relevant_col, relevant_col2 = Task_cols[args.task]

    # df = df[df['WorkerId'].isin(select_workers(df, relevant_col))]
    # breakpoint()
    # time_analysis = analyze_perHitTime(df[[
    #     'WorkerId', 'WorkTimeInSeconds',
    #     'Answer.clickedConsentTime',
    #     'Answer.clickedSubmitTime']])

    # print(time_analysis)

    #### TEMPORARY FIX: remove 1 in the column names of the dataframe
    df = df.rename(columns={i: i.replace("1", "") for i in df.columns})
    df_stats = stats_of_interest(df[relevant_col])
    if args.suggestion:
        df_stats_2 = stats_of_interest(df[relevant_col2], has_text=True)

    # Assign special NaN value
    if args.binary:
        # Special treatment for the offensiveness rating
        df["Answer.offensivenessRating"] = [
            args.boundary + 1 if j == "not offensive" and i == -1 else i
            for i, j in zip(
                df["Answer.offensivenessRating"],
                render_offensiveness_labels(df["Input.offensiveness"]),
            )
        ]
        df[relevant_col] = df[relevant_col].replace(-1, np.nan)
        df[relevant_col] = (df[relevant_col] > args.boundary).astype(int)
    else:
        # df[relevant_col] = df[relevant_col].replace(-1, np.nan)
        df[relevant_col] = df[relevant_col].replace(-1, 1.5)

    # The current quality calculation process will binarize the rating automatically.
    df_quality = output_quality_ratio(df, relevant_col, args)
    df[relevant_col] = normalize_df(df[relevant_col])
    df_agreement = calculate_agreement(df, relevant_col, args)

    df_final = pd.concat([df_quality, df_agreement], join="outer")

    # make the format of the table better
    df_stats = pretty(df_stats)
    if args.suggestion:
        df_stats_2 = pretty(df_stats_2)
    df_final = pretty(df_final)

    df_stats.to_csv(args.output_folder + "/" + "stats.csv")
    if args.suggestion:
        df_stats_2.to_csv(args.output_folder + "/" + "stats_2.csv")
    df_final.to_csv(args.output_folder + "/" + "quality.csv")


if __name__ == "__main__":
    main()
