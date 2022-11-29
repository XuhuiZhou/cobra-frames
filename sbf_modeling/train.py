import os
from typing import Sequence

import gin
from absl import app, flags, logging
from datasets.arrow_dataset import Dataset

from sbf_modeling import BaseSBFModel, gin_utils

_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

FLAGS = flags.FLAGS


def train(
    *,
    model: BaseSBFModel,
    train_data: Dataset,
    model_dir: str,
):
    logging.info("Training model")
    model = model.train(train_data, save_model_dir=model_dir)
    logging.info("Model trained")


def main(_):
    # Create gin-configurable version of `train`.
    train_using_gin = gin.configurable(train)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )

    train_using_gin()  # type: ignore


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
