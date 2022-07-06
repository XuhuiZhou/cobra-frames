import pandas as pd
import numpy as np

from collections import Counter
from itertools import combinations

def distNumeric(l1,l2):
  return np.abs(l1-l2)

def distJaccard(s1,s2):
  s1 = set(s1)
  s2 = set(s2)
  
  if len(s1) == len(s2) and len(s1) == 0:
    # Jaccard similarity is 1: distance is 0
    return 0

  return 1 - (len(s1 & s2) / len(s1 | s2))

def computePairwiseAgreement(df,valCol,groupCol="HITId",minN=2,distF=distNumeric):
  """Computes pairwise agreement.
  valCol: the column with the answers (e.g., Lickert scale values)
  groupCol: the column identifying the rated item (e.g., HITId, post Id, etc)
  """
  g = df.groupby(groupCol)[valCol]
  ppas = {}
  n = 0
  for s, votes in g:
    if len(votes) >= minN:
      pa = np.mean([1-distF(*v) for v in combinations(votes,r=2)])
      ppas[s] = pa
      n += 1
      if pd.isnull(pa):
        print("Pairwise agreement is null for group "+g)
        # embed()
    # else: print(len(votes))
  ppa = np.mean(list(ppas.values()))
  
  if pd.isnull(ppa):
    print(f"Pairwise agreement probs for column {valCol}")
    # embed()
    
  return ppa, n, pd.Series(ppas)

def computeRandomAgreement(df,valCol,groupCol="HITId",distF=distNumeric):
  distrib = Counter(df[valCol])
  agree = 0.
  tot = 0.
  i = 0
  for p1 in distrib:
    for p2 in distrib:
      a1 = p1
      a2 = p2
      num,denom = 1-distF(a1,a2), 1
      if p1 == p2:
        agree += distrib[p1]*(distrib[p2]-1) *num/denom
        tot += distrib[p1]*(distrib[p2]-1)
      else:
        agree += distrib[p1]*(distrib[p2]) *num/denom
        tot += distrib[p1]*distrib[p2]
    i += 1
  return agree/tot

def computeAlpha(df,valCol,groupCol="HITId",minN=2,distF=distNumeric):
  """Computes Krippendorf's Alpha"""
  d = df[~df[valCol].isnull()]
  ppa, n, groups = computePairwiseAgreement(
    d,valCol,groupCol=groupCol,minN=minN,distF=distF)

  d2 = d[d[groupCol].isin(groups.index)]
  
  # Only computing random agreement on HITs that
  # we computed pairwise agreement for.
  if len(groups):
    rnd = computeRandomAgreement(d2,valCol,groupCol=groupCol,distF=distF)
    
    # Skew: computes how skewed the answers are; Krippendorf's Alpha
    # behaves terribly under skewed distributions.
    if d2[valCol].dtype == float or d2[valCol].dtype == int:
      skew = d2[valCol].mean()
    else:
      if isinstance(d2[valCol].iloc[0],list) or isinstance(d2[valCol].iloc[0],set):
        skew = 0
      else:
        skew = d2[valCol].describe()["freq"] / len(d2)
  else:
    rnd = np.nan
    skew = 0
    
  alpha = 1 - ((1-ppa)/(1-rnd))
  
  return dict(alpha=alpha, ppa=ppa, rnd_ppa=rnd, skew=skew, n=n)


if __name__ == "__main__":
  # creates fake data
  # 5-point lickert scale
  # rater1 is normal rater
  rater1 = pd.Series(np.random.randint(0,5,100))
  
  # rater2 agrees with rater1 most of the time
  rater2 = np.random.uniform(size=rater1.shape)
  rater2 = pd.Series((rater2 > .1).astype(int) * rater1 )

  # rater3 should be random
  rater3 = pd.Series(np.random.randint(0,5,100))

  df = pd.DataFrame([rater1,rater2,rater3],index=["r1","r2","r3"]).T
  df.index.name = "id"
  df = df.reset_index()
  longDf = df.melt(
    id_vars=["id"], value_vars=["r1","r2","r3"],
    var_name="raterId", value_name="rating")
  longDf["ratingBinary"] = (longDf["rating"]/longDf["rating"].max()).round(0)
  

  # metrics = computeKappa(longDf)
  ppa = computePairwiseAgreement(longDf,"ratingBinary",groupCol="id")
  rndPpa = computeRandomAgreement(longDf,"ratingBinary",groupCol="id")
  scores = computeAlpha(longDf,"ratingBinary",groupCol="id")