import pandas as pd
import json

def tag_posts(file_name, source_file_name):
    df = pd.read_csv(file_name)
    posts = df['Input.statement'].to_list()
    df_source = pd.read_csv(source_file_name)
    source_posts = df_source['statement'].to_list()
    source_dict = {
        'sbic': source_posts[:40],
        'mAgr': source_posts[40:]
    } 
    tags = []
    for i in posts:
        for key in source_dict:
            if i in source_dict[key]:
                tags.append(key)
    assert len(tags) == len(posts)
    df['source_tag'] = tags
    return df

file_name = './data/mturk/CSBF-Verification-Pilot-2-incomplete.csv'
source_file_name = './data/mturk/mturk.pilotTwo.mAgr_and_sbic.csv'
df = tag_posts(file_name, source_file_name)
df.to_csv(file_name)

