from collections import ChainMap
from typing import Any, Callable, Dict, List, Sequence, Tuple, cast

import evaluate
import nltk
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer
from transformers.trainer import EvalPrediction


def postprocess_text(
    tokenizer: PreTrainedTokenizer, eval_preds: EvalPrediction
) -> Tuple[List[str], List[str]]:
    preds, labels = eval_preds
    if not isinstance(preds, np.ndarray):
        preds = preds[0]
    assert isinstance(tokenizer.pad_token_id, int)
    preds = np.where(labels != -100, preds, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def aggregated_metrics(
    metrics: Sequence[Callable[[EvalPrediction], Dict[str, Any]]],
    eval_preds: EvalPrediction,
) -> Dict[str, Any]:
    return dict(ChainMap(*map(lambda m: m(eval_preds), metrics)))


def aggregated_metrics_with_postprocess(
    metrics: Sequence[Callable[[List[str], List[str]], Dict[str, Any]]],
    tokenizer: PreTrainedTokenizer,
    eval_preds: EvalPrediction,
) -> Dict[str, Any]:
    preds, labels = postprocess_text(tokenizer, eval_preds)
    return dict(ChainMap(*map(lambda m: m(preds, labels), metrics)))


def prediction_metrics(
    number_of_examples: int, preds: List[str], labels: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Print out the predictions and labels for a few examples.
    args:
        tokenizer: The tokenizer used to decode the predictions.
        number_of_examples: The number of examples to print out.
        eval_preds: The predictions and labels.
    returns:
        A dictionary with the following keys
            - predictions: A dataframe with the predictions and labels.
    """
    predictions = preds[:number_of_examples]
    decoded_labels = labels[:number_of_examples]
    sample_df = pd.DataFrame.from_dict(
        {"predictions": predictions, "labels": decoded_labels}
    )
    return {"predictions": sample_df}


def bleu_metrics(preds: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Compute the BLEU score.
    args:
        preds: The predictions.
        labels: The labels.
    returns:
        A dictionary with the following keys
            - bleu: The BLEU score.
    """
    bleu = evaluate.load("bleu")
    result = cast(
        Dict[str, float],
        bleu.compute(predictions=preds, references=labels),
    )
    prediction_lens = [len(pred.split()) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result
