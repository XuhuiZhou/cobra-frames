import os
from typing import Callable, Sequence, cast

import gin
import pandas as pd
import torch
from absl import app, flags, logging
from datasets.arrow_dataset import Dataset
from transformers import Seq2SeqTrainingArguments

from sbf_modeling import BaseSBFModel, ExplainModel, gin_utils
from sbf_modeling.evaluation_utils import generic_evaluate_function

_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

FLAGS = flags.FLAGS

import sys
import traceback


class Suppressor(object):
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            # Do normal exception handling
            pass

    def write(self, x):
        pass


@gin.configurable
def evaluate(
    *,
    prediction_dict: dict,
    reference_dict: dict,
):
    for key in prediction_dict:
        logging.info(f"Evaluating {key}")
        with Suppressor():
            prediction = prediction_dict[key]
            reference = reference_dict[key]
            assert len(prediction) == len(reference)
            results = generic_evaluate_function(
                ["bleu", "rouge", "meteor", "bertscore", "mauve"],
                prediction,
                reference,
            )
        logging.info(f"Results for {key}: {results}")


def predict(
    *,
    model: BaseSBFModel,
    output_dir: str,
):
    logging.info("Model predicting")
    if torch.cuda.is_available():
        model.model = model.model.cuda()  # type: ignore
    answer_dict = model.predict()
    logging.info("Model inference done")
    answer_df = pd.DataFrame.from_dict(answer_dict)
    answer_df.to_csv(os.path.join(output_dir, "answer.csv"), index=False)
    logging.info("Evaluating predictions")
    evaluate(prediction_dict=answer_dict)  # type: ignore


def main(_):
    # Create gin-configurable version of `train`.
    predict_using_gin = gin.configurable(predict)
    predict_using_gin = cast(Callable[[], None], predict_using_gin)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )

    predict_using_gin()


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "gin_file",
        default=None,
        help="Path to gin configuration file. Multiple paths may be passed and "
        "will be imported in the given order, with later configurations  "
        "overriding earlier ones.",
    )

    flags.DEFINE_multi_string(
        "gin_bindings", default=[], help="Individual gin bindings."
    )

    flags.DEFINE_list(
        "gin_search_paths",
        default=["."],
        help="Comma-separated list of gin config path prefixes to be prepended "
        "to suffixes given via `--gin_file`. If a file appears in. Only the "
        "first prefix that produces a valid path for each suffix will be "
        "used.",
    )

    gin_utils.run(main)
