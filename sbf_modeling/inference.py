import os
from typing import Sequence

import gin
import pandas as pd
import torch
from absl import app, flags, logging
from datasets.arrow_dataset import Dataset
from transformers import Seq2SeqTrainingArguments

from sbf_modeling import BaseSBFModel, ExplainModel, gin_utils

_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

FLAGS = flags.FLAGS


def predict(
    *,
    model_dir: str,
    test_data: Dataset,
    output_dir: str,
    per_device_predict_batch_size: int = 16,
):
    logging.info("Model predicting")
    model = ExplainModel(model_dir, from_local=True)
    if torch.cuda.is_available():
        model.model = model.model.cuda()  # type: ignore
    answer_dict = model.predict(
        test_data,
        args=Seq2SeqTrainingArguments(
            output_dir=".log/_explain_model",
            per_device_eval_batch_size=per_device_predict_batch_size,
            predict_with_generate=True,  # generation in evaluation
            prediction_loss_only=False,  # generation in evaluation
        ),
    )
    logging.info("Model inference done")
    answer_df = pd.DataFrame.from_dict(answer_dict)
    answer_df.to_csv(os.path.join(output_dir, "answer.csv"), index=False)


def main(_):
    # Create gin-configurable version of `train`.
    predict_using_gin = gin.configurable(predict)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )

    predict_using_gin()  # type: ignore


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
