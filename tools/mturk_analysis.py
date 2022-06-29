import argparse
import re
import simpledorff
import pandas as pd
import numpy as np
from collections import Counter
#from disagree import metrics

def count_scale(df):
    scale = [0, 0]
    values = df.item.to_list()
    for i in values:
        scale[int(i>3)]+=1
    return scale

def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)

def calculate_agreement_fleiss(file_name):
    df = pd.read_csv(file_name)
    df = df[['HITId', 'hg', 'clear', 'likely', 'related', 'stance']]
    for i in ['clear', 'likely', 'related', 'stance']:
        new_df = df[['HITId', 'hg', i]]
        new_df = new_df.rename(columns={i: "item"})
        new_df = new_df.groupby(by=["HITId", "hg"]).apply(count_scale)
        matrix = [i for i in new_df]
        matrix = np.array(matrix)
        agreement = fleiss_kappa(matrix)
        print(f"{i}'s agreement is {agreement}")


def transform_table(df):
    """transform the table to form using the worker id as the column names.
    row: HITID->unqiue id of each instance.
    col: worker id
    """
    workers = set(df.WorkerId.to_list())
    table_dict = {}
    for i in workers:
        work_df = df[df.WorkerId==i]
        table_dict[i] = {}
        for index, row in work_df.iterrows():
            table_dict[i][row[0]] = row[-1] # row[0] is the HITID, row[-1] is the rating
    new_df = pd.DataFrame(table_dict)
    #new_df = new_df.reindex(range(len(new_df)))
    return new_df


def joint_o_probability(ann1, ann2):
    zipped = [(i,j) for i,j in zip(ann1, ann2) if i!=-1 and j!=-1]
    if len(zipped)==0:
        return None
    agree = [1 if label[0] == label[1] else 0 for label in zipped]
    return sum(agree) / len(agree)


def output_quality_ratio(df, col, bar):
    df = df.groupby(by=["HITId"]).sum()
    df = (df>bar)
    sum_ = df.sum()
    total = len(df)
    frames = {}
    for i in col:
        frames[i] = [f'{100*sum_[i]/total:.2f}%']
    df = pd.DataFrame.from_dict(frames)
    df = df.rename(index={0:"quality"})
    return df
 

def calculate_agreement_pairwise(df, col):
    frames = {}
    for i in col:
        new_df = df[['HITId', 'WorkerId', i]]
        new_df = new_df.rename(columns={i: "item"})
        new_df = transform_table(new_df)
        k = new_df.corr(method=joint_o_probability)
        pairwise = k.sum().sum()/k.count().sum()
        #k.to_csv(f'pairwise_matrix_{i}.csv')
        #print(k)
        frames[i] = [f'{pairwise*100:.2f}%']
    df = pd.DataFrame.from_dict(frames)
    df = df.rename(index={0:"pairwise agreement"})
    return df

def calculate_agreement_krip(df, col):
    labels = [0,1]
    for i in col:
        new_df = df[['HITId', 'hg', 'WorkerId', i]]
        new_df['id'] = df['HITId'] + df['hg']
        alpha = simpledorff.calculate_krippendorffs_alpha_for_df(new_df,experiment_col='id',
                                                 annotator_col='WorkerId',
                                                 class_col=i)
        print(f"{i}'s Krippendorff's alpha: {alpha:.2f}")

def stats_of_interest(df, has_text=False):
    """
    Customized function to calculate the distribution stats of the table
    """
    if has_text:
        frames = {}
        l = len(df)
        for i in df.keys():
            num_has_text = l- df[i].str.contains('{}|none').sum()
            frames[i] = [num_has_text]
        stats = pd.DataFrame.from_dict(frames)
        return stats

    else:
        frames = []
        for i in df.keys():
            frames.append(df[i].value_counts())
        stats = pd.concat(frames, join="outer", axis=1)
        stats = stats.rename(index={
            3: "very likely", 
            1: "somewhat unlikely", 
            2: "somewhat likely",
            0: "very unlikely",
            -1: "none"
            })
        return stats

def pretty(df):
    """
    Customized function to make the table better readable
    """
    keys = df.keys()
    new_keys = [i.split('.')[-1] for i in keys]
    df = df.rename(columns={i:j for i,j in zip(keys, new_keys)})
    return df

def main():
    """
    Script for analyzing Mturk produced data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_folder")
    parser.add_argument("--bar", default=1, type=int)
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    df = df.rename(columns={
        "Answer.targetGroupEffectSuggestion": "Answer.targetGroupCogReactSuggestion", 
        "Answer.targetGroupReactionSuggestion": "Answer.targetGroupEmoReactSuggestion"
    })
    #df_stats = stats_of_interest(df)

    relevant_col = [
        'Answer.targetGroupRating',
        'Answer.intentRating', 
        'Answer.implicationRating',
        'Answer.offensivenessRating',
        'Answer.powerDiffRating',
        'Answer.targetGroupEmoReactRating',
        'Answer.targetGroupCogReactRating',
    ]
    relevant_col2 = [
        'Answer.targetGroupSuggestion',
        'Answer.intentSuggestion', 
        'Answer.implicationSuggestion',
        'Answer.offensivenessSuggestion',
        'Answer.powerDiffSuggestion',
        'Answer.targetGroupEmoReactSuggestion',
        'Answer.targetGroupCogReactSuggestion',
    ]
    df_stats = stats_of_interest(df[relevant_col])
    df_stats_2 = stats_of_interest(df[relevant_col2], has_text=True)
    df[relevant_col] = (df[relevant_col]>1).astype(int)
    df_quality = output_quality_ratio(df, relevant_col, args.bar)
    df_pairwise_agr = calculate_agreement_pairwise(df, relevant_col) 

    #df_krip = calculate_agreement_krip(df, relevant_col)
    df_final = pd.concat([
        df_quality, df_pairwise_agr
    ], join="outer") 

    #make the format of the table better    
    df_stats = pretty(df_stats)
    df_stats_2 = pretty(df_stats_2)
    df_final = pretty(df_final)

    df_stats.to_csv(args.output_folder+'/'+'stats.csv')
    df_stats_2.to_csv(args.output_folder+'/'+'stats_2.csv')
    df_final.to_csv(args.output_folder+'/'+'quality.csv')

    #calculate_agreement_krip(df, relevant_col)
    


if __name__ == '__main__':
    main()