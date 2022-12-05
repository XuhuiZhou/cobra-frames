import pandas as pd


def main():
    # read the human-scored qualifcation files and analyze them
    df_scored_1 = pd.read_csv(
        "data/mturk/qualification/qual3_results_humanscored.csv"
    )
    df_scored_2 = pd.read_csv(
        "data/mturk/qualification/qual4_results_humanscored.csv"
    )
    df_scored = pd.concat([df_scored_1, df_scored_2])
    df_scored = df_scored[
        [
            "WorkerId",
            "Keep?",
            "Answer.annotatorAge",
            "Answer.annotatorGender",
            "Answer.annotatorMinority",
            "Answer.annotatorPolitics",
            "Answer.annotatorRace",
        ]
    ]
    print(
        "the number of workers who have been scored is {}".format(
            len(df_scored)
        )
    )
    df_scored_keep = df_scored[df_scored["Keep?"] == 1.0]
    print(
        "the number of workers who have been scored and kept is {}".format(
            len(df_scored_keep)
        )
    )
    print(df_scored_keep.describe())
    print(df_scored_keep["Answer.annotatorAge"].value_counts())
    print(df_scored_keep["Answer.annotatorGender"].value_counts())
    print(df_scored_keep["Answer.annotatorRace"].value_counts())
    print(df_scored_keep["Answer.annotatorMinority"].value_counts())
    print(df_scored_keep["Answer.annotatorPolitics"].value_counts())


if __name__ == "__main__":
    main()
