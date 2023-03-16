import os

import boto3
import pandas as pd


def count(row):
    # count the number of suggestions
    count = 0
    for r in row.keys():
        if "Suggestion" in r:
            suggestion = row[r]
            if len(suggestion) > 8:
                count += 1
    return count


folder = "data/mturk/for_bonus/"
# read all the csv files
dfs = [pd.read_csv(folder + f) for f in os.listdir(folder)]
workers = {}
for df in dfs:
    for index, row in df.iterrows():
        if row["WorkerId"] in workers:
            workers[row["WorkerId"]][1] += count(row)
        else:
            workers[row["WorkerId"]] = [row["AssignmentId"], count(row)]

# sum up the number of suggestions
num_suggestions = sum([v[1] for v in workers.values()])
print("Total number of suggestions: {}".format(num_suggestions))

mturk = boto3.session.Session(profile_name="msap-cs-uw").client(
    "mturk", region_name="us-east-1"
)
for worker in workers:
    if (
        worker
        not in [
            "A27PVIL93ZMY46",
            "A2T11H7YI7QPGD",
            "APRZ7BR8C0ZMQ",
            "A345TDMHP3DQ3G",
            "A3GWRDHAURRNK6",
            "A748N8WFG2HR4",
            "A23KAJRDVCVGOE",
            "A32W24TWSWXW",
            "A37GOI3N77WX21",
        ]
        and workers[worker][1] > 0
    ):
        print(
            "Worker {} has {} suggestions".format(worker, workers[worker][1])
        )
        amount = workers[worker][1] * 0.05
        response = mturk.send_bonus(
            WorkerId=worker,
            BonusAmount=f"{amount:.2f}",
            AssignmentId=workers[worker][0],
            Reason="Thank you for your suggestions! We are sorry for the delay in paying the bonus since we were waiting for the final results of the experiment. We pay you a bonus of $0.05 for each suggestion you made. Note that we have a system that checks the quality of your suggestions, and we only assign the bonus if your suggestions are of high quality. Also, note that we pay in total to avoid repetitive email notifications. Therefore, we only use one of the assignment IDs assigned to you throughout this study. In general, you have done an amazing job and it is our pleasure to work with you!",
        )
        print(response)
