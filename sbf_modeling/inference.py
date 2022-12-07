import os
from typing import Sequence

import gin
import pandas as pd
from absl import app, flags, logging
from datasets.arrow_dataset import Dataset

from sbf_modeling import BaseSBFModel, gin_utils

_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

FLAGS = flags.FLAGS


def predict(
    *,
    model: BaseSBFModel,
    test_data: Dataset,
    model_dir: str,
):
    logging.info("Model predicting")
    answer_dict = model.predict(test_data, model_dir)
    logging.info("Model inference done")
    answer_df = pd.DataFrame.from_dict(answer_dict)
    answer_df.to_csv(os.path.join(model_dir, "answer.csv"), index=False)


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
