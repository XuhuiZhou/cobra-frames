from dis import Instruction
from email.policy import default
from operator import index
from random import seed
from turtle import st
import numpy as np
import pandas as pd
import json, time
import argparse
import openai

from IPython import embed
from tqdm import tqdm

# openai.api_key = "sk-5sqlr5GGEcAf33RBvOnDFnzp547YNQPJHUNpSOYK" # UW key
# openai.api_key = "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" # Mosaic key
# openai.api_key = "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" # CMU key
openai.api_key = "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"

tox_condition_dict = {
    "sbic.trn.r60.gpt3socCont.csv": ['hasBiasedImplication', 1],
    "SBIC.v2.agg.trn.csv": ['hasBiasedImplication', 1],
    "mAgr.r60.gpt3socCont.csv": None,
    "mAgr.onlyQuotes.csv": None,
}

variables = [
    # 'conversationContext',
    'statement',
    'speechContext',
    'speakerIdentity',
    'listenerIdentity',
    # 'speakerListenerRelationship',
    'intent',
    'targetGroup',
    'relevantPowerDynamics',
    'implication',
    'targetGroupEmotionalReaction',
    'targetGroupCognitiveReaction',
    'offensiveness',
]

formatting = {
    'speechContext': "[Context of statement] {}[/]",
    'speakerIdentity': "[Speaker identity/characteristics] {}[/]",
    'listenerIdentity': "[Listener identity/characteristics] {}[/]",
    # 'speakerListenerRelationship': "Speaker-listener relationship: {}",
    'relevantPowerDynamics': "[Relevant power dynamics] {}[/]",
    'conversationContext': '[Conversational context] {}[/]',
    'statement': '[Statement] {}[/]',
    'intent': '[Intent] {}[/]',
    'offensiveness': '[Offensiveness] {}[/]',
    'targetGroup': '[Targeted/referenced minority group] {}[/]',
    'implication': '[Implied meaning/stereotype] {}[/]',
    'targetGroupEmotionalReaction': '[Targeted minority group emotional reaction] {}[/]',
    'targetGroupCognitiveReaction': '[Targeted minority group cognitive reaction] {}[/]',
}

revFormat = {v.replace(" {}[/]", ""): k for k, v in formatting.items()}

#instructions = "Given a statement or a conversational snippet, explain the identities and power differentials of the speaker/listener, the intent behind the statement, why it might be offensive, and which minority group it targets. \n\n"
instructions = "Given a statement or a conversational snippet, \
think in what situation could the statement happen, \
who will be the potential speakers, \
and who will be the potential listeners. Diverse outputs are encouraged.\n\n"


class GPT3_Sbf():
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

    def getGPT3prob(self, x, variant="text-davinci-002", attempt=0):
        time.sleep(0.06)
        try:
            r = openai.Completion.create(
                engine=variant,
                prompt=x,
                temperature=self.temp,
                stop=self.stop,
                n=self.generated_num,
                # best_of=4,
                # top_p=0.5,
                max_tokens=225,
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
                return self.getGPT3prob(x, variant=variant, attempt=attempt + 1)

        # text = '[/]'.join([c['text'] for c in r["choices"]])
        # text = text.split('[/]')
        # text = '[//]'.join(text)
        # text = text+'[/]'
        # TODO: fix this
        gens = []
        for c in r["choices"]:
            logprobs = c["logprobs"]["token_logprobs"]
            text = c["text"]
            # out = pd.Series(dict(socialContextGPT3=text,socialContextGPT3logprob=np.sum(logprobs)))
            totalEx = x.split("\n\n")[-1] + text
            d = self.parseGPT3output(totalEx)

            # d = {k: l.replace(v.replace(" {}",""),"").strip() for k, v in formatting.items() for l in totalEx.split("\n") if v.replace(" {}","") in l}
            d["logprob"] = np.sum(logprobs)
            gens.append(pd.Series(d))
        return gens

    def parseGPT3output(self, t):
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
                sample(min(len(examples), n), random_state=self.args.random_seed).iterrows()
            ]
        )
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

    def sbflise_table(self, posts):
        # Table preprocess
        posts = posts.rename(
            columns={
                "actual_quote": "post",
                "text": "post"
            }
        )  #Rename the columns for extraction
        posts["examples"] = posts[args.statement_col].apply(
            self.addExamplesToPost, examples=self.examples, n=self.args.n_examples
        )
        fPosts = posts.apply(self.formatPosts, **self.args.__dict__, preamble=instructions, axis=1)
        tqdm.pandas(ascii=True)

        out = fPosts.progress_apply(self.getGPT3prob)
        list_of_out = []
        for i in out:
            list_of_out += i
        out = pd.DataFrame(list_of_out)
        cols = [c for c in self.variables if c in out.columns]
        out = out[cols]
        out = out.set_index(pd.Index([i // 3 for i in out.index]))
        posts = posts.reset_index()
        #posts = pd.concat([posts, out], axis=1)
        posts = posts.join(out, how='right')
        return posts


class GPT3_Sbf_spk(GPT3_Sbf):
    """
    A text completion class based on the demonstrations, but only limits to generate speakers
    """
    def __init__(self, args, examples) -> None:
        super().__init__(args, examples)
        self.variables = variables[:2]
        self.generated_num = 3
        self.temp = 1.2
        self.stop = "[/]"


class GPT3_Sbf_context(GPT3_Sbf):
    """
    A text completion class based on the demonstrations, but only limits to generate speakers
    """
    def __init__(self, args, examples) -> None:
        super().__init__(args, examples)
        self.variables = variables[:4]
        self.generated_num = 1
        self.args.n_examples = 6
        self.temp = 0.8
        #self.stop = ""


def main(args):
    examples = pd.read_csv(args.example_file)
    examples_context = pd.read_csv(args.example_file_context)
    posts = pd.read_csv(args.input_file, index_col=0)  # Read the csv file without index column
    # Select desired toxic class statements; need to be further refined!
    if args.tox_class == 'toxic':
        condition = tox_condition_dict[(args.input_file).split('/')[-1]]
        if condition:
            posts = posts[posts[condition[0]] == condition[1]]

    if args.sample:
        posts = posts.sample(args.sample, random_state=args.random_seed)

    # Initialize the GPT3 Gen class
    # gpt3_gen = GPT3_Sbf_spk(args, examples)
    # posts = gpt3_gen.sbflise_table(posts)

    gpt3_gen = GPT3_Sbf_context(args, examples_context)
    posts = gpt3_gen.sbflise_table(posts)

    try:
        del posts["post"]
    except:
        pass

    try:
        del posts["socialContextGPT3"]
        del posts["socialContextGPT3logprob"]
    except:
        pass
    posts.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file")
    p.add_argument("--example_file", default="./data/promptExamples.v2.csv", type=str)
    p.add_argument("--example_file_context", default="./data/examples.v2.contextonly.csv", type=str)
    p.add_argument("--statement_col", default="post")
    p.add_argument("--conversationContext_col", default="conversationContext")
    p.add_argument("--n_examples", default=7, type=int)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_file")
    p.add_argument("--tox_class", default="toxic", type=str)
    args = p.parse_args()
    main(args)
