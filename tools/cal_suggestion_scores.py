import pandas as pd
import sys
from bert_score import score
from collections import defaultdict

def score_rule(i,j,v):
    """
    The rule to calculate the score.
    The score can come from one reference or multiple references.
    """
    print("---------------------------sdfsdgsdf")
    if type(j)==str:
        P, R, F1 = score([i], [j], lang='en', rescale_with_baseline=True)
    else:
        P, R, F1 = score([i], j, lang="en", rescale_with_baseline=True)
    return F1[0].item()

def attention_check_filter(df):
    """
    Remove the annotators that fail the attention check 
    and print out the failed numbers.
    """
    att1 = df['Answer.attentionCheck1']==df['Answer.attentionCheckCorrect1']
    att2 = df['Answer.attentionCheck2']==df['Answer.attentionCheckCorrect2']
    att3 = df['Answer.attentionCheck3']==df['Answer.attentionCheckCorrect3'] 
    att_pass = att1 & att2 & att3
    att_pass = [int(i) for i in att_pass.to_list()]
    print(f"There are {sum(att_pass)} annotators passing the attention check.")
    return att_pass

def compose_suggestions(df, suggestions_var):
    worker_IDs = df['WorkerId'].to_list()
    suggestions = defaultdict(list)
    for s_var in suggestions_var:
        for id, s in zip(worker_IDs, df[s_var].to_list()):
            suggestions[id].append(s)
    return suggestions

def obtain_reference(df, s_var):
    s_var_unique = list(set([i.split('.')[1][:-11] for i in s_var])) # From Answer.xxxSuggestion2 to xxx
    reference_dict = {}
    for i in s_var_unique:
        values = df[i].to_list()
        for index, v in enumerate(values):
            if index%2==1:
                reference_dict['Answer.'+i+'Suggestion'+str(index//2+1)] = \
                    "" if len(v)==1 else (v.split('(')[1]).strip(')')
    # Reorder reference:
    reference = []
    for i in s_var:
        reference.append(reference_dict[i])
    return reference

def calculate_scores(suggestions, reference, suggestions_var):
    scores = []
    for id in suggestions:
        worker_score = []
        worker_suggestions = suggestions[id]
        for i,j,v in zip(worker_suggestions, reference, suggestions_var):
            # If the reference is None, we don't calculate the score.
            if j!="":
                worker_score.append(score_rule(i,j,v))
        scores.append(sum(worker_score)/len(worker_score))
    return scores, suggestions.keys()

qual_file = sys.argv[1] if len(sys.argv) > 1 else './data/mturk/qual3_results.csv' 
scored_file = sys.argv[2] if len(sys.argv) > 2 else './data/mturk/qual3_results_scored.csv'
answer_file = sys.argv[3] if len(sys.argv) > 3 else './data/mturk/mturk_qual_answer_key.csv'

df = pd.read_csv(qual_file)
#att_pass = attention_check_filter(df)
df_answer = pd.read_csv(answer_file)

suggestions_var = [i for i in df.keys() if 'Suggestion' in i]
worker_suggestions = compose_suggestions(df, suggestions_var)
suggestions_reference = obtain_reference(df_answer, suggestions_var)
scores, workers = calculate_scores(worker_suggestions, suggestions_reference, suggestions_var)

# Process the annotators' scores with attention checks
# scores = [i*j for i,j in zip(scores, att_pass)]
df['WorkerSugestionScore'] = scores
df.to_csv(scored_file)
print("finish assigning suggestion scores to the workers")