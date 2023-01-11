import argparse

import pandas as pd


def parse_offensiveness(df):
    # TODO calculate GPT-3 generated contexts with humans' alignment rate
    offensive = []
    moreOff = []
    for i, j, k, s in zip(
        df["Answer.harmlessOne"],
        df["Answer.offensiveOne"],
        df["Answer.moreOff"],
        df["Input.statement"],
    ):
        # check if the offensiveness is consistent
        if (i == j and (i == "Scenario A" or i == "Scenario B")) or (
            i == k and k != "0"
        ):
            print("error")
            print(i, j, k)
            print(s)
            offensive += [None, None]
            moreOff += [None, None]
        elif i == "Scenario A" and j == "Scenario B" and k == "Scenario B":
            offensive += [0, 1]
            moreOff += [0, 1]
        elif i == "Scenario B" and j == "Scenario A" and k == "Scenario A":
            offensive += [1, 0]
            moreOff += [1, 0]
        elif k == "Scenario A":
            offensive += [-1, -1]
            moreOff += [1, 0]
        elif k == "Scenario B":
            offensive += [-1, -1]
            moreOff += [0, 1]
        else:
            offensive += [-1, -1]
            moreOff += [-1, -1]
    return offensive, moreOff


def get_new_column_name(columns):
    col_names = {}
    for i in columns:
        if i.startswith("Input."):
            col_names[i] = i.replace("Input.", "").strip("2")
        elif i.startswith("Answer."):
            col_names[i] = i.replace("Answer.", "").strip("2")
        else:
            col_names[i] = i
    return col_names


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/mturk/advContexts/annotation_summary.csv",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/mturk/advContexts/advContexts.csv",
    )
    parser.add_argument("--pre_output_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="majority")

    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    columns = df.columns
    offensive, moreOff = parse_offensiveness(df)
    col_1 = [
        col
        for col in columns
        if col
        not in [
            "Input.situationalContext2",
            "Input.speaker2",
            "Input.listener2",
        ]
    ]
    col_2 = [
        col
        for col in columns
        if col
        not in ["Input.situationalContext", "Input.speaker", "Input.listener"]
    ]
    df_1 = df[col_1]
    df_2 = df[col_2]
    col_name_1 = get_new_column_name(df_1.columns)
    col_name_2 = get_new_column_name(df_2.columns)
    df_1.rename(columns=col_name_1, inplace=True)
    df_1.index = df_1.index * 2

    df_2.rename(columns=col_name_2, inplace=True)
    df_2.index = df_2.index * 2 + 1

    if args.mode == "majority":
        df_overall = pd.concat([df_1, df_2]).sort_index()
    elif args.mode == "allapp" and args.pre_output_file is not None:
        df_overall = pd.read_csv(args.pre_output_file)

    df_overall[f"offensive_{args.mode}"] = offensive
    df_overall[f"moreOff_{args.mode}"] = moreOff
    print(len(df_overall))
    # drop nan value
    if args.mode == "allapp" and args.pre_output_file is not None:
        print("drop nan")
        df_overall.dropna(inplace=True)
        print(len(df_overall))
    df_overall.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
