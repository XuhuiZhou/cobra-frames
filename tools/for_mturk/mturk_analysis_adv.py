from tools.for_mturk.mturk_analysis import *
from tools.for_mturk.mturk_analysis_sen import analyze_perCategory

category_map = {
    1: "Scenario B",
    0: "Scenario A",
    -1: "Both/neither",
}


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

    # df = df[df['WorkerId'].isin(select_workers(df, relevant_col))]
    # breakpoint()
    # time_analysis = analyze_perHitTime(df[[
    #     'WorkerId', 'WorkTimeInSeconds',
    #     'Answer.clickedConsentTime',
    #     'Answer.clickedSubmitTime']])

    # print(time_analysis)

    #### TEMPORARY FIX: remove 1 in the column names of the dataframe

    relevant_col, relevant_col2 = Task_cols[args.task]
    df = df.rename(columns={i: i.replace("1", "") for i in df.columns})
    df_ratings = None
    df_final = None

    for i in [0, 1]:
        df_info, df_final_cate = analyze_perCategory(df, relevant_col, i, args)
        if args.record_annotation_summary:
            df_info = record_annotation_summary(df_info, df, args)
        if df_final is None:
            df_final = df_final_cate.rename(
                index={"approval rate": category_map[i]}
            )
        else:
            # insert a new row
            df_final = df_final.append(
                df_final_cate.rename(index={"approval rate": category_map[i]})
            )
        if df_ratings is None:
            # Copy the columns with ratings to a new dataframe
            df_ratings = df_info[relevant_col].copy()
            for j in relevant_col:
                df_ratings[j] = df_info[j].replace(1, category_map[i])
        else:
            for j in relevant_col:
                df_ratings[j][df_info[j] == 1] = df_info[j][
                    df_info[j] == 1
                ].replace(1, category_map[i])
    df_info[relevant_col] = df_ratings
    df_info.to_csv(f"{args.output_folder}/annotation_summary.csv", index=False)
    df_final.to_csv(args.output_folder + "/" + "quality.csv")


if __name__ == "__main__":
    main()
