import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from IPython import embed
from tqdm import tqdm

np.random.seed(345)
random.seed(345)

import openai

openai.api_key = "sk-5sqlr5GGEcAf33RBvOnDFnzp547YNQPJHUNpSOYK"  # UW key
# openai.api_key = "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" # Mosaic key

instructions = (
    "Provide the social context in which a statement was probably said.\n\n"
)

variables = ["conversationContext", "statement", "socialContext"]

formatting = {
    "socialContext": "Social context: {}",
    "conversationContext": "Conversational context: {}",
    "statement": "Statement: {}",
    "intent": "Intent: {}",
    "offensiveness": "Offensiveness: {}",
    "targetGroup": "Target/reference group: {}",
    "implication": "Implication/stereotype: {}",
    "targetGroupReaction": "Target group reaction: {}",
}


def getGPT3prob(x, variant="text-davinci-001", attempt=0):
    time.sleep(0.06)
    try:
        r = openai.Completion.create(
            engine=variant,
            prompt=x,
            temperature=0.3,
            stop="\n",
            # n=3,
            # best_of=5,
            # top_p=0.5,
            max_tokens=60,
            logprobs=1,
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
            return getGPT3prob(x, variant=variant, attempt=attempt + 1)

    c = r["choices"][0]
    text = c["text"]
    logprobs = c["logprobs"]["token_logprobs"]
    out = pd.Series(
        dict(socialContextGPT3=text, socialContextGPT3logprob=np.sum(logprobs))
    )

    return out


def formatPosts(p, preamble):
    out = preamble + "\n\nStatement: " + p + "\nSocial context:"
    return out


def main(args):
    exampleFile = "promptExamples.v0.json"
    examples = json.load(open(exampleFile))
    formattedExamples = "\n\n".join(
        [
            "\n".join([formatting[v].format(e[v]) for v in variables if e[v]])
            for e in examples["examples"]
        ]
    )

    posts = pd.read_csv(args.input_file)
    if args.debug:
        posts = posts.sample(args.debug)

    fPosts = posts[args.text_col].apply(
        formatPosts, preamble=instructions + formattedExamples
    )

    tqdm.pandas(ascii=True)
    out = fPosts.progress_apply(getGPT3prob)
    posts = pd.concat([posts, out], axis=1)

    posts.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file")
    p.add_argument("--text_col", default="post")
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--output_file")
    args = p.parse_args()
    main(args)
