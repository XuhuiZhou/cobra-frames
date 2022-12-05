import os
import string
import sys

import pandas as pd


def combine_toxigen(path, range):
    """
    Combine all the chunks of toxigen data into one csv file
    """
    df = pd.DataFrame()
    paths = os.listdir(path)
    for i in paths:
        if i.endswith(".csv") and int(i.split("_")[1].strip(".csv")) in range:
            df = pd.concat([df, pd.read_csv(os.path.join(path, i))])
    return df, paths


def clean_string(s):
    """
    Clean the string for Mturk annotation
    """
    printable = set(string.printable)
    s = "".join(filter(lambda x: x in printable, s))
    return s


def clean_for_mturk(df):
    """
    Clean the data for Mturk annotation
    CSV file should be UTF-8 encoded and cannot contain characters with encodings larger than 3 bytes.
    """
    for i in df.columns:
        if df[i].dtype == "object":
            df[i] = df[i].apply(clean_string)
    return df


def main():
    inference_toxigen_path = "data/inference_data/toxigen_shuffled"
    folder = sys.argv[1]
    range_start = int(sys.argv[2])
    range_end = int(sys.argv[3])
    output_folder = os.path.join(inference_toxigen_path, folder)
    # if the directory does not exist, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df, paths = combine_toxigen(
        inference_toxigen_path, range(range_start, range_end)
    )
    df_yes = df[df["statementCheck"] == "yes"]
    df_complete = df_yes.dropna()
    # sample 500 rows from the complete data as the Mturk annotation data
    df_mturk = df_complete.sample(n=500, random_state=1)
    df_mturk = clean_for_mturk(df_mturk)

    # save the paths file name into a txt file
    with open(os.path.join(output_folder, "paths.txt"), "w") as f:
        for i in paths:
            f.write(i + "\n")
    df.to_csv(output_folder + "/toxigen.csv", index=False)
    df_complete.to_csv(output_folder + "/toxigen_complete.csv", index=False)
    df_mturk.to_csv(output_folder + "/toxigen_mturk.csv", index=False)
    # output the number of rows
    print("Number of rows: {}".format(len(df)))
    # output the number of rows with statementCheck == 'yes'
    print(
        "Number of rows with statementCheck == 'yes': {}".format(len(df_yes))
    )
    # output the number of rows with statementCheck == 'yes' and no missing values
    print(
        "Number of rows with statementCheck == 'yes' and no missing values: {}".format(
            len(df_complete)
        )
    )
    print("Done combining")


if __name__ == "__main__":
    main()
