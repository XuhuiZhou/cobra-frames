import pandas as pd

def contain_suspicious_token(sentence):
    if sentence[-1] in [',', ';', ':', '<', '(', '-', '[', '{']:
        return True
    else:
        return False

def complete(generation):
    is_complete = []
    for i in generation:
        if contain_suspicious_token(i):
            is_complete.append(False)
        else:
            is_complete.append(True)
    return is_complete

def chunk_dataframe(df):
    """
    Split dataframe into chunks of 1000 rows
    And save them into a folder
    """
    chunk_size = 100
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df[i:i+chunk_size])
    
    for i, df_chunk in enumerate(chunks):
        df_chunk.to_csv('data/cleaned_data/toxigen/toxigen_{}.csv'.format(i), index=False)

def chunk_and_shuffle(df):
    """
    Split dataframe into chunks of 1000 rows
    And save them into a folder
    """
    chunk_size = 30
    chunks = []
    # shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    for i in range(0, len(df), chunk_size):
        chunks.append(df[i:i+chunk_size])
    
    for i, df_chunk in enumerate(chunks):
        df_chunk.to_csv('data/cleaned_data/toxigen_shuffled/toxigen_{}.csv'.format(i), index=False)

# Read the data
df = pd.read_csv('data/source_data/toxigen.csv')
df = df[df['prompt_label'] == 1]
is_complete = complete(df['generation']) 
df = df[is_complete]
df.to_csv('data/cleaned_data/toxigen.csv', index=False)
print("Done cleaning")
chunk_dataframe(df)
print("Done chunking")
chunk_and_shuffle(df)
print("Done chunking and shuffling")