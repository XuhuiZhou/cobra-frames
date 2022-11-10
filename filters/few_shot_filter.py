import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from dis import Instruction
from email.policy import default
from operator import index
from random import seed
from turtle import st
import numpy as np
import pandas as pd
import warnings
import json, time
import argparse
import openai

from IPython import embed
from tqdm import tqdm

#os.environ['TRANSFORMERS_CACHE'] = '/home/xuhuiz/projects/xuhuiz/context-sbf/hf_home'
#os.environ['TRANSFORMERS_CACHE'] = '/projects/tir4/users/xuhuiz/context-sbf/hf_home/'
#os.environ['TRANSFORMERS_CACHE'] = '/projects/tir3/users/xuhuiz/socialiq/'

variables = [
    'statement',
    'speechContext',
    'speakerIdentity',
    'listenerIdentity',
]

formatting = {
    'speechContext': "[Context of statement] {}[/]",
    'speakerIdentity': "[Speaker identity/characteristics] {}[/]",
    'listenerIdentity': "[Listener identity/characteristics] {}[/]",
    'statement': '[Statement] {}[/]',
    'situationQua': '[Plausibility and informativeness of the situational context] {}[/]',
    'speakerQua': '[Plausibility of the speaker] {}[/]',
    'listenerQua': '[Plausibility of the listener] {}[/]',
}

revFormat = {v.replace(" {}[/]", ""): k for k, v in formatting.items()}

#instructions = "Given a statement or a conversational snippet, explain the identities and power differentials of the speaker/listener, the intent behind the statement, why it might be offensive, and which minority group it targets. \n\n"
instructions = "Given a statement or a conversational snippet, \
generate in what situation could the statement happen, \
who will be the potential speakers, \
and who will be the potential listeners.\n\n"


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def tokenize_function(examples):
    breakpoint()
    examples['text'] = f"Statement: {}"
    examples['text'] = text_prompt + examples['text']
    return tokenizer(examples["question"], examples['answer'], padding="max_length", truncation=True)

class T5_filter():
    """
    A text completion class based on the demonstrations 
    """
    def __init__(self, args, examples) -> None:
        self.args = args
        self.examples = examples
        self.variables = variables
        self.variables_full = variables
        self.generated_num = 1
        self.example_size = 10
        self.temp = 0.3
        self.stop = "\n\n"
        self.logprobs = 1

    def getT5generation(self, x, variant="text-davinci-002", attempt=0):
        time.sleep(0.06)
        try:
            r = openai.Completion.create(
                engine=variant,
                prompt=x,
                temperature=self.temp,
                stop=self.stop,
                n=1,
                # best_of=4,
                top_p=0.9,
                max_tokens=225,
                presence_penalty=0.5,
                frequency_penalty = 0.5,
                logprobs=self.logprobs,
            )
        except openai.error.APIError as e:
            print(e)
            print("Sleeping for 10 seconds")
            time.sleep(10)
            if attempt > 10:
                print("Reached attempt limit, giving up")
                return None
            else:
                print("Trying again")
                return self.getT5generation(x, variant=variant, attempt=attempt + 1)

        c = r["choices"][0]
        text = c["text"]
        logprobs = c["logprobs"]["token_logprobs"]
        # out = pd.Series(dict(socialContextGPT3=text,socialContextGPT3logprob=np.sum(logprobs)))
        totalEx = x.split("\n\n")[-1]+text
        d = self.parse_output(totalEx)

        # d = {k: l.replace(v.replace(" {}",""),"").strip() for k, v in formatting.items() for l in totalEx.split("\n") if v.replace(" {}","") in l}
        d["logprob"] = np.sum(logprobs)

        return pd.Series(d)

    def parseOutput(self, t):
        fields = [f.strip() for f in t.split("[/]") if f]
        out = {k: f.replace(p, "").strip() for p, k in revFormat.items() for f in fields if p in f}
        return out

    def addExamplesToPost(self, p, examples, n=10):
        formattedExamples = "\n\n".join(
            [
                "\n".join(
                    [
                        formatting[v].format(e[v])
                        for v in self.variables if e[v] and not pd.isnull(e[v])
                    ]
                ) for ix, e in examples.
                sample(min(len(examples), n)).iterrows()
            ]
        )
        if self.negative_examples:
            formattedExamples += ("\n\n"+self.negative_examples)
        return formattedExamples

    def formatPosts(self, p, preamble="", **kwargs):
        # out = preamble+"\n\nStatement: "+p+"\nSocial context:"
        #f = [formatting[v].format(e[v]) for v in variables if e[v]]
        f = [
            formatting[v].format(p[kwargs[v + "_col"]]) for v in self.variables
            if v + "_col" in kwargs and kwargs[v + "_col"] in p and p[kwargs[v + "_col"]] != ""
        ]
        out = preamble + p["examples"] + "\n\n" + "\n".join(f) + "\n"
        out += formatting[self.variables[self.variables.index("statement") +
                                         1]].split("{}")[0].strip()
        return out
    
    def expand_table(self, posts):
        posts = posts.loc[posts.index.repeat(self.generated_num)]
        posts = posts.reset_index(drop=True)
        return posts

    def sbflise_table(self, posts):
        # Table preprocess
        posts = posts.rename(
            columns={
                "actual_quote": "post",
                "text": "post",
                "generation": "post"
            }
        )  #Rename the columns for extraction
        # Expand table
        posts = self.expand_table(posts)
        posts["examples"] = posts[args.statement_col].apply(
            self.addExamplesToPost, examples=self.examples, n=self.args.n_examples
        )
        fPosts = posts.apply(self.formatPosts,**args.__dict__,preamble=instructions,axis=1)
        tqdm.pandas(ascii=True)
                
        #out = fPosts.progress_apply(self.getT5generation)
        out = self.get
        cols = [c for c in self.variables if c in out.columns]
        out = out[cols]
        posts = pd.concat([posts,out],axis=1)
        return posts

file_name = './data/cache/annotation_summary.csv'

dataset = load_dataset("csv", data_files = file_name)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
dataloader = DataLoader(tokenized_datasets, batch_size=8)

model.eval()
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model.generate(**batch)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))