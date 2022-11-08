# IPython log file

import pandas as pd
df = pd.read_csv("data/dynaHate/DynaHate.v0.2.3.csv",index_col=0)
n = 1000

df["hasTarget"] = "yes"
df.loc[df["target"]=="none","hasTarget"] = "none"
df.loc[df["target"]=="notgiven","hasTarget"] = "notgiven"

out = []
for ix,c in df[df.split=="train"].groupby("hasTarget"):
    out.append(c.sample(min(len(c),n)))
outDf = pd.concat(out)

outDf.to_csv("dynaHate.trn.r3k.strat.csv",index=False)

