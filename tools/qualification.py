import argparse
import sys
from IPython import embed
import pandas as pd
from tqdm import tqdm
import boto3

# this will require having an .aws/credentials file on your
mturk = boto3.session.Session(profile_name='msap-cs-uw').client("mturk",region_name='us-east-1')
# mturk = boto3.client('mturk')

def main(args):
  dfs = [pd.read_csv(f) for f in args.qual_score_files]
  df = pd.concat(dfs).set_index("WorkerId")
  
  print(df[args.qual_score_col].value_counts().sort_index())

  if not args.qual_id:
    return
  
  for wId, r in tqdm(df.iterrows(),ascii=True,total=len(df)):
    if args.qual_score_col:
      # MTurk only allows an integer qual score between 0 and 100
      s = int(100*r[args.qual_score_col])
    else:
      s = 1
    mturk.associate_qualification_with_worker(
      QualificationTypeId=args.qual_id,
      WorkerId=wId,
      IntegerValue=s,
      SendNotification=False)



if __name__=="__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--qual_score_files",nargs="+")
  p.add_argument("--qual_id")
  p.add_argument("--qual_score_col")
  args = p.parse_args()
  print(args)
  main(args)