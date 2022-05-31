import numpy as np
import pandas as pd
from IPython import embed
import pandas as pd
import json, time
import argparse
from tqdm import tqdm
import openai
openai.api_key = "sk-5sqlr5GGEcAf33RBvOnDFnzp547YNQPJHUNpSOYK" # UW key
# openai.api_key = "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" # Mosaic key

variables = [
    'socialContext', 'conversationContext', 'statement', 'intent', 
    'offensiveness', 'targetGroup', 'implication', 'targetGroupReaction']
formatting = {
    'socialContext': 'Social context: {}',
    'conversationContext': 'Conversational context: {}',
    'statement': 'Statement: {}',
    'intent': 'Intent: {}',
    'offensiveness': 'Offensiveness: {}',
    'targetGroup': 'Targeted/referenced minority group: {}',
    'implication': 'Implication/stereotype: {}',
    'targetGroupReaction': 'Targeted minority group reaction: {}'}

instructions = "From a US sociocultural perspective, explain the intent behind a statement, why it might be offensive, and which minority group it targets."

def getGPT3prob(x,variant="text-davinci-001",attempt=0):
  time.sleep(0.06)
  try:
    r = openai.Completion.create(
      engine=variant,
      prompt=x,
      temperature=0.3,
      stop="\n\n",
      # n=3,
      # best_of=5,
      # top_p=0.5,
      max_tokens=160,
      logprobs=1,
    )
  except openai.error.APIError as e:
    print(e)
    print("Sleeping for 10 seconds")
    time.sleep(10)
    if attempt>10:
      print("Reached attempt limit, giving up")
      return None
    else:
      print("Trying again")
      return getGPT3prob(x,variant=variant,attempt=attempt+1)
    

  c = r["choices"][0]
  text = c["text"]
  logprobs = c["logprobs"]["token_logprobs"]
  # out = pd.Series(dict(socialContextGPT3=text,socialContextGPT3logprob=np.sum(logprobs)))
  totalEx = x.split("\n\n")[-1]+text
  d = {k: l.replace(v.replace(" {}",""),"").strip() for k, v in formatting.items() for l in totalEx.split("\n") if v.replace(" {}","") in l}
  d["logprob"] = np.sum(logprobs)
  return pd.Series(d)

def formatPosts(p,preamble="",**kwargs):
  # out = preamble+"\n\nStatement: "+p+"\nSocial context:"
  #f = [formatting[v].format(e[v]) for v in variables if e[v]]
  f = [formatting[v].format(p[kwargs[v+"_col"]])
       for v in variables if v+"_col" in kwargs and kwargs[v+"_col"] in p and p[kwargs[v+"_col"]] != ""]
  out = preamble+"\n\n"+"\n".join(f)+"\nIntent:"

  return out


def main(args):
  exampleFile = "promptExamples.v0.json"
  examples = json.load(open(exampleFile))
  formattedExamples = "\n\n".join([
    "\n".join([formatting[v].format(e[v]) for v in variables if e[v]])
    for e in examples["examples"]
  ])
  
  posts = pd.read_csv(args.input_file)
  if args.debug:
    posts = posts.sample(args.debug)

  posts = posts.rename(columns={"actual_quote": "post","text": "post"})
    
  fPosts = posts.apply(formatPosts,**args.__dict__,preamble=instructions+formattedExamples,axis=1)

  tqdm.pandas(ascii=True)
  out = fPosts.progress_apply(getGPT3prob)
  posts = pd.concat([posts,out],axis=1)
  # embed();exit()  
  posts.to_csv(args.output_file,index=False)
  

if __name__ =="__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--input_file")
  p.add_argument("--statement_col",default="post")
  p.add_argument("--socialContext_col",default="socialContextGPT3")
  p.add_argument("--conversationContext_col",default="conversationContext")
  p.add_argument("--debug",type=int,default=0)
  p.add_argument("--output_file")
  args = p.parse_args()
  main(args)


