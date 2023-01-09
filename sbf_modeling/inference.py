import os
from typing import Callable, List, Sequence, cast

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


@gin.configurable
def evaluate(
    *,
    prediction_dict: dict,
    reference_dict: dict = {},
    result_dump_path: str = "",
    metric_names: List[str] = ["bleu"],
):
    logged_results = {}
    for key in prediction_dict:
        logging.info(f"Evaluating {key}")
        prediction = prediction_dict[key]
        reference = reference_dict[key]
        assert len(prediction) == len(reference)
        results = generic_evaluate_function(
            metric_names,
            prediction,
            reference,
        )
        logged_results[key] = results
        logging.info(f"Results for {key}: {results}")
    if result_dump_path:
        with open(result_dump_path, "w") as f:
            f.write("," + ",".join(metric_names) + "\n")
            for key in logged_results:
                f.write(
                    f"{key},"
                    + ",".join(
                        [
                            str(logged_results[key][metric])
                            for metric in metric_names
                        ]
                    )
                    + "\n"
                )
            f.write(
                "average,"
                + ",".join(
                    [
                        str(
                            sum(
                                [
                                    logged_results[key][metric]
                                    for key in logged_results
                                ]
                            )
                            / len(logged_results)
                        )
                        for metric in metric_names
                    ]
                )
                + "\n"
            )


def predict(
    *,
    model: BaseSBFModel,
    output_dir: str,
):
    # Check if the output file already exists
    if os.path.exists(os.path.join(output_dir, "answer.csv")):
        logging.info("Output file already exists, skipping inference")
        answer_df = pd.read_csv(os.path.join(output_dir, "answer.csv"))
        answer_dict = answer_df.to_dict(orient="list")
    else:
        logging.info("Model predicting")
        if torch.cuda.is_available():
            model.model = model.model.cuda()  # type: ignore
        answer_dict = model.predict()
        logging.info("Model inference done")
        answer_df = pd.DataFrame.from_dict(answer_dict)
        # check whether output dir exists, if not make it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        answer_df.to_csv(os.path.join(output_dir, "answer.csv"), index=False)
    logging.info("Evaluating predictions")
    evaluate(prediction_dict=answer_dict)


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
