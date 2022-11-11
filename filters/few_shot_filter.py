import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from dis import Instruction
from email.policy import default
from operator import index
from random import seed
from turtle import st
import warnings
import json, time
import argparse

from IPython import embed
from tqdm import tqdm

#os.environ['TRANSFORMERS_CACHE'] = '/home/xuhuiz/projects/xuhuiz/context-sbf/hf_home'
#os.environ['TRANSFORMERS_CACHE'] = '/projects/tir4/users/xuhuiz/context-sbf/hf_home/'
#os.environ['TRANSFORMERS_CACHE'] = '/projects/tir3/users/xuhuiz/socialiq/'

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class T5_filter():
    """
    A text completion class based on the demonstrations 
    """
    def __init__(self, args, examples) -> None:
        self.variables = [
            'statement',
            'speechContext',
            'speakerIdentity',
            'listenerIdentity',
            'situationRating',
            'speakerIdenRating',
            'listenerIdenRating',
        ]

        self.formatting = {
            'speechContext': "[Situational context of statement] {}[/]",
            'speakerIdentity': "[Speaker identity/characteristics] {}[/]",
            'listenerIdentity': "[Listener identity/characteristics] {}[/]",
            'statement': '[Statement] {}[/]',
            'situationRating': '[Informativeness of the situational context] {}[/]',
            'speakerIdenRating': '[Plausibility of the speaker] {}[/]',
            'listenerIdenRating': '[Plausibility of the listener] {}[/]',
        }

        self.revFormat = {v.replace(" {}[/]", ""): k for k, v in self.formatting.items()}

        #self.instructions = "Given a statement or a conversational snippet, explain the identities and power differentials of the speaker/listener, the intent behind the statement, why it might be offensive, and which minority group it targets. \n\n"
        self.instructions = "Given a statement or a conversational snippet, \
        as well as its situational context, speaker, and listener identity, \
        generate the rating of those scenarios. \n\n"
        self.args = args
        self.end_variable = "listenerIdentity"
        self.examples = examples
        self.generated_num = 1
        self.example_size = 10
        self.temp = 0.3
        self.stop = "\n\n"
        self.logprobs = 1

    def getT5generation(self, df):
        # Convert series to dataframe
        df = pd.DataFrame({'text': df})
        dataset = Dataset.from_pandas(df)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")
        dataloader = DataLoader(tokenized_datasets, batch_size=4)
        model.to(device)
        model.eval()
        text_output = []
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model.generate(**batch)
                print((self.tokenizer.batch_decode(outputs, skip_special_tokens=True)))
                text_output += (self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return text_output
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def parseOutput(self, out):
        # Map yes to 1, no to 0
        parsed_output = []
        for i in out:
            if 'yes' in i:
                parsed_output.append(1)
            else:
                parsed_output.append(0)
        return parsed_output

    def addExamplesToPost(self, p, examples, n=10):
        formattedExamples = "\n\n".join(
            [
                "\n".join(
                    [
                        self.formatting[v].format(e[v])
                        for v in self.variables if e[v] and not pd.isnull(e[v])
                    ]
                ) for ix, e in examples.
                sample(min(len(examples), n)).iterrows()
            ]
        )
        return formattedExamples

    def formatPosts(self, p, preamble="", **kwargs):
        # out = preamble+"\n\nStatement: "+p+"\nSocial context:"
        #f = [self.formatting[v].format(e[v]) for v in variables if e[v]]
        f = [
            self.formatting[v].format(p[kwargs[v + "_col"]]) for v in self.variables
            if v + "_col" in kwargs and kwargs[v + "_col"] in p and p[kwargs[v + "_col"]] != ""
        ]
        out = preamble + p["examples"] + "\n\n" + "\n".join(f) + "\n"
        out += self.formatting[self.variables[self.variables.index(self.end_variable) +
                                         1]].split("{}")[0].strip()
        return out

    def calculate_scores(self, output, target):
        acc = (output == target).sum() / len(target)
        precision = (output * target).sum() / output.sum()
        recall = (output * target).sum() / target.sum()
        f1 = 2 * precision * recall / (precision + recall)
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}

    def generate_table(self, posts):
        # Table preprocess
        posts = posts.rename(
            columns={
                "Input.statement": "statement",
                "Input.speechContext": "speechContext",
                "Input.speakerIdentity": "speakerIdentity",
                "Input.listenerIdentity": "listenerIdentity",
            }
        )  # Rename the columns for extraction
        # Expand table
        posts["examples"] = posts['statement'].apply(
            self.addExamplesToPost, examples=self.examples, n=self.args.n_examples
        )
        fPosts = posts.apply(self.formatPosts,**args.__dict__,preamble=self.instructions,axis=1)
        tqdm.pandas(ascii=True)
                
        #out = fPosts.progress_apply(self.getT5generation)
        out = self.getT5generation(fPosts)
        out = self.parseOutput(out)
        posts[self.variable_of_interest] = out
        return posts

class T5_filter_speaker(T5_filter):
    """
    A text filter class based on the demonstrations for speaker"""

    def __init__(self, args, examples) -> None:
        super().__init__(args, examples)
        self.variables = [
            'statement',
            'speakerIdentity',
            'speakerIdenRating',
        ]
        # self.formatting = {
        #     'speechContext': "[Situational context of statement] {}[/]",
        #     'speakerIdentity': "As well as the potential speaker: {}",
        #     'listenerIdentity': "[Listener identity/characteristics] {}[/]",
        #     'statement': 'Given the statement: {}',
        #     'speakerIdenRating': 'So the answer is: {}[/]',
        #     'listenerIdenRating': '[Plausibility of the listener] {}[/]',
        # }
        self.generated_num = 1
        self.end_variable = "speakerIdentity"
        self.variable_of_interest = "speakerIdenRating_model"
        self.example_size = 10
        self.temp = 0.3
        self.stop = "\n\n"
        self.logprobs = 1

def main(args):
    examples = pd.read_csv(args.example_file)
    posts = pd.read_csv(args.input_file)  # Read the csv file without index column

    if args.sample:
        posts = posts.sample(args.sample, random_state=args.random_seed)

    #t5_gen = T5_filter(args, examples)
    t5_gen = T5_filter_speaker(args, examples)
    # see whether the output file exists
    if os.path.exists(args.output_file):
        print("Output file already exists, please delete it first.")
        posts = pd.read_csv(args.output_file)
    else:
        posts = t5_gen.generate_table(posts)
        posts.to_csv(args.output_file, index=False)

    scores = t5_gen.calculate_scores(posts['speakerIdenRating_model'], posts['Answer.speakerIdenRating'])
    print(scores)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", default="./data/cache/annotation_summary.csv")
    p.add_argument("--example_file", default="./data/prompts/examples.t5.contextCritic_Anno.csv", type=str)
    #p.add_argument("--example_file", default="./data/prompts/examples.t5.contextCritic_AnnoPlusEx.csv", type=str)
    p.add_argument("--statement_col", default="statement")
    p.add_argument("--speechContext_col", default="speechContext")
    p.add_argument("--speakerIdentity_col", default="speakerIdentity")
    p.add_argument("--listenerIdentity_col", default="listenerIdentity")
    p.add_argument("--n_examples", default=7, type=int)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_file", default="./data/cache/annotation_summary.t5.csv")
    args = p.parse_args()
    main(args)