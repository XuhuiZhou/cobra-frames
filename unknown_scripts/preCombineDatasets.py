import json

import pandas as pd
from IPython import embed

# dynaHate
dynaHate = pd.read_csv(
    "/home/maartens/data/dynaHate/DynaHate.v0.2.3.csv", index_col=0
)
print("DynaHate")
print(
    (dynaHate["label"] + "-" + dynaHate["type"] + "-" + dynaHate["target"])
    .value_counts()
    .head(10)
)
print("...\n")


# ContextAbuseData
def aggregateThreads(x):
    embed()


def findTitlesForPosts(df):
    posts = df["dataType"] == "post"
    df.loc[posts, "postTitle"] = df.loc[posts, "info_order"].str.replace(
        "-post", "-title"
    )
    df = df.merge(
        df.drop_duplicates("info_id")[["info_order", "id", "meta_text"]],
        right_on="info_order",
        left_on="postTitle",
        suffixes=("", "_title"),
        how="left",
    )
    return df


def findParentsForComments(df):
    dfNoDup = df.drop_duplicates("info_id")
    comms = df["dataType"] == "comment"
    commsWithAncestors = df

    anc = 1
    commsWithAncestors = commsWithAncestors.merge(
        dfNoDup[
            ["info_id", "info_order", "id", "meta_text", "info_id.parent"]
        ],
        left_on="info_id.parent",
        right_on="info_id",
        suffixes=("", f"_anc{anc}"),
        how="left",
    )
    while not commsWithAncestors[f"meta_text_anc{anc}"].isnull().all():
        anc += 1
        print(anc)
        commsWithAncestors = commsWithAncestors.merge(
            dfNoDup[
                ["info_id", "info_order", "id", "meta_text", "info_id.parent"]
            ],
            left_on=f"info_id.parent_anc{anc-1}",
            right_on="info_id",
            suffixes=("", f"_anc{anc}"),
            how="left",
        )
    embed()
    commsWithParents = df.merge(
        dfNoDup[
            ["info_id", "info_order", "id", "meta_text", "info_id.parent"]
        ],
        right_on="info_id",
        left_on="info_id.parent",
        suffixes=("", "_parent"),
        how="left",
    )

    commsWithGrandParents = commsWithParents.merge(
        dfNoDup[
            ["info_id", "info_order", "id", "meta_text", "info_id.parent"]
        ],
        left_on="info_id.parent_parent",
        right_on="info_id",
        suffixes=("", "_grandparent"),
        how="left",
    )
    commsWithGreatGrandParents = commsWithGrandParents.merge(
        dfNoDup[
            ["info_id", "info_order", "id", "meta_text", "info_id.parent"]
        ],
        left_on="info_id.parent_grandparent",
        right_on="info_id",
        suffixes=("", "_greatgrandparent"),
        how="left",
    )

    embed()
    exit()


print("Context Abuse Data")
contAbuseFull = pd.read_csv(
    "/home/maartens/data/contextualAbuseData/cad_v1_1.tsv", sep="\t"
)
contAbuseFull["dataType"] = "comment"
contAbuseFull.loc[
    contAbuseFull["info_order"].str.contains("post"), "dataType"
] = "post"
contAbuseFull.loc[
    contAbuseFull["info_order"].str.contains("title"), "dataType"
] = "title"
print(contAbuseFull["dataType"].value_counts())
contAbuseFull = findTitlesForPosts(contAbuseFull)
findParentsForComments(contAbuseFull)
# contAbuseFull.groupby("info_id.link").apply(aggregateThreads)
# contAbuse_trn = pd.read_csv("/home/maartens/data/contextualAbuseData/cad_v1_1_train.tsv",sep="\t")
# contAbuse_tst = pd.read_csv("/home/maartens/data/contextualAbuseData/cad_v1_1_test.tsv",sep="\t")
# contAbuse_dev = pd.read_csv("/home/maartens/data/contextualAbuseData/cad_v1_1_dev.tsv",sep="\t")

# print(contAbuseFull["labels"].value_counts())
print()

# Malevdialogues
malDial = pd.read_csv(
    "/home/maartens/data/malevDialogs/mdrdc_dataset_response_context_rewrite_6k.tsv",
    sep="\t",
)
malDialLabels = pd.read_csv(
    "/home/maartens/data/malevDialogs/label_name.txt",
    sep="\t",
    names=["label", "labelName"],
)
malDial = malDial.merge(malDialLabels, on="label").sort_values(
    ["Number", "turn"]
)
print(malDial["labelName"].value_counts())
print()
embed()
