# Contextual Social Bias Frames repo

## Getting started
* All the dataset for this repo is located in `/home/xuhuiz/projects/xuhuiz/context-sbf/data` now.
* The source code is built aroud PyTorch, and has the following main dependencies:

    - Python 3.8
    - PyTorch 1.12.1
    - transformers 4.22.2

    For more extensive dependencies, see `requirements.txt`.

        pip install -r requirements.txt
* We use `pre-commit` hooks to unify the code style, please refer to https://pre-commit.com/ for installation and usage.


## Generating contexts
Generate contexts using the bash script:
`populateSBFwithGPT3_large_scale.sh`

## Data cartography
Work on things in the `cartography` folder

## Generating contexts and explanations with GPT3
Main file for generating CSFB with GPT-3 is `populateSBFwithGPT3.v2.py`
- requires updating/filling in the [examples spreadsheet](https://docs.google.com/spreadsheets/d/1y3WwnVPdgM_hP2lZVE9L3IVy9Tpq5QNMWPVsVGMxX6g/edit#gid=1334320915), download it as `examples.v2.csv`
- might have to update the column names and/or the variables [populateSBFwithGPT3.v2.py#L13](populateSBFwithGPT3.v2.py#L13) and formatting [populateSBFwithGPT3.v2.py#L23](populateSBFwithGPT3.v2.py#L23) variables in the code
- then run the command with the default parameters with each of the files below as the `--input_file`


Input data files:
- `dynaHate.trn.r60.gpt3socCont.csv` : 60 randomly sampled examples from DynaHate
- `implHate.noSap.r60.gpt3socCont.csv` : 60 randomly sampled examples from Implicit Hate Corpus (by Caleb Ziems et al 2022) -- excluding SBIC portions
- `mAgr.r60.gpt3socCont.csv` : 60 randomly sampled examples from a data dump of microaggressions.com - `sbic.trn.r60.gpt3socCont.csv` : 60 randomly sampled examples from SBIC


Notes:
- The `populateSBFwithGPT3.v2.py` script generates the CSBF (I think?)
- The `populateSBFwithGPT3.v5.py` script generates the context
- It uses my hardcoded CMU API key, which way run out of money LOL. Switch to UW or AI2 one but ask me before?
