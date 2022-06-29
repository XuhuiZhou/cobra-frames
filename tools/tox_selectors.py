import pandas as pd
import json

def select_sbic(args):
    df = pd.read_csv(args.file_name)
    df_tox = df[df['hasBiasedImplication']==1]
    df_tox = df_tox.sample(frac=args.fraction, random_state=args.random_seed)
    df_tox.to_csv('')