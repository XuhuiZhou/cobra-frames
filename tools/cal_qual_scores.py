import pandas as pd
from collections import defaultdict

def score_rule(i,j):
    if i==-1:
        return 0
    elif i<=1 and j<=1:
        return 1
    elif i>1 and j>1:
        return 1
    else:
        return 0

def compose_ratings(df, ratings_var):
    worker_IDs = df['WorkerId'].to_list()
    ratings = defaultdict(list)
    for r_var in ratings_var:
        for id, r in zip(worker_IDs, df[r_var].to_list()):
            ratings[id].append(r)
    return ratings

def obtain_reference(df, r_var):
    r_var_unique = list(set([i.split('.')[1][:-7] for i in r_var]))
    reference_dict = {}
    for i in r_var_unique:
        values = df[i].to_list()
        for index, v in enumerate(values):
            if index%2==1:
                reference_dict['Answer.'+i+'Rating'+str(index//2+1)] = \
                    int(v) if len(v)==1 else int(v.split(' ')[0])
    # Reorder reference:
    reference = []
    for i in r_var:
        reference.append(reference_dict[i])
    return reference

def calculate_scores(ratings, reference):
    scores = []
    for id in ratings:
        worker_score = []
        worker_ratings = ratings[id]
        for i,j in zip(worker_ratings, reference):
            worker_score.append(score_rule(i,j))
        scores.append(sum(worker_score)/len(worker_score))
    return scores, ratings.keys()

qual_file = './data/mturk/qual1_results.csv' 
answer_file = './data/mturk/mturk_qual_answer_key.csv'

df = pd.read_csv(qual_file)
df_answer = pd.read_csv(answer_file)

ratings_var = [i for i in df.keys() if 'Rating' in i]
worker_ratings = compose_ratings(df, ratings_var)
ratings_reference = obtain_reference(df_answer, ratings_var)
scores, workers = calculate_scores(worker_ratings, ratings_reference) 
df['WorkerScores'] = scores

df.to_csv(qual_file)
print("finish assigning scores to the workers")

