import sys
from collections import defaultdict

import pandas as pd


def score_rule(i, j, v):
    if v in [
        "Answer.powerDiffRating1",
        "Answer.powerDiffRating2",
        "Answer.powerDiffRating3",
    ]:
        if i == 0 or i == 1 or i == -1:
            return 1
    if i == -1:
        return 0
    elif i <= 1 and j <= 1:
        return 1
    elif i > 1 and j > 1:
        return 1
    else:
        return 0


def attention_check_filter(df):
    """
    Remove the annotators that fail the attention check
    and print out the failed numbers.
    """
    att1 = df["Answer.attentionCheck1"] == df["Answer.attentionCheckCorrect1"]
    att2 = df["Answer.attentionCheck2"] == df["Answer.attentionCheckCorrect2"]
    att3 = df["Answer.attentionCheck3"] == df["Answer.attentionCheckCorrect3"]
    att_pass = att1 & att2 & att3
    att_pass = [int(i) for i in att_pass.to_list()]
    print(f"There are {sum(att_pass)} annotators passing the attention check.")
    return att_pass


def compose_ratings(df, ratings_var):
    worker_IDs = df["WorkerId"].to_list()
    ratings = defaultdict(list)
    for r_var in ratings_var:
        for id, r in zip(worker_IDs, df[r_var].to_list()):
            ratings[id].append(r)
    return ratings


def obtain_reference(df, r_var):
    r_var_unique = list(set([i.split(".")[1][:-7] for i in r_var]))
    reference_dict = {}
    for i in r_var_unique:
        values = df[i].to_list()
        for index, v in enumerate(values):
            if index % 2 == 1:
                reference_dict[
                    "Answer." + i + "Rating" + str(index // 2 + 1)
                ] = (int(v) if len(v) == 1 else int(v.split(" ")[0]))
    # Reorder reference:
    reference = []
    for i in r_var:
        reference.append(reference_dict[i])
    return reference


def calculate_scores(ratings, reference, ratings_var):
    scores = []
    for id in ratings:
        worker_score = []
        worker_ratings = ratings[id]
        for i, j, v in zip(worker_ratings, reference, ratings_var):
            worker_score.append(score_rule(i, j, v))
        scores.append(sum(worker_score) / len(worker_score))
    return scores, ratings.keys()


qual_file = (
    sys.argv[1] if len(sys.argv) > 1 else "./data/mturk/qual3_results.csv"
)
scored_file = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "./data/mturk/qual3_results_scored.csv"
)
answer_file = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "./data/mturk/mturk_qual_answer_key.csv"
)

df = pd.read_csv(qual_file)
att_pass = attention_check_filter(df)
df_answer = pd.read_csv(answer_file)

ratings_var = [i for i in df.keys() if "Rating" in i]
worker_ratings = compose_ratings(df, ratings_var)
ratings_reference = obtain_reference(df_answer, ratings_var)
scores, workers = calculate_scores(
    worker_ratings, ratings_reference, ratings_var
)

# Process the annotators' scores with attention checks
scores = [i * j for i, j in zip(scores, att_pass)]
df["WorkerScores"] = scores

df.to_csv(scored_file)
print("finish assigning scores to the workers")
