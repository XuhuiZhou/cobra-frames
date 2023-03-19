import argparse

import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def binarize_offensiveness(offensive):
    offensive = [
        "none" if x is None or type(x) == float else x for x in offensive
    ]  # TODO: further investigate why this is necessary
    offensive = list(map(lambda x: x.lstrip().lower(), offensive))
    offensive = list(map(lambda x: x.replace("very ", ""), offensive))
    offensive = list(map(lambda x: x.replace("highly ", ""), offensive))
    offensive = list(map(lambda x: x.replace("extremely ", ""), offensive))

    for i, off in enumerate(offensive):
        if "anti-sem" in off:
            offensive[i] = "anti-semitic"
        if (
            "none" in off
            or "not offensive" in off
            or "not necessarily offensive" in off
            or "n/a" in off
            or "not applicable" in off
            or "harmless" in off
        ):
            offensive[i] = "not offensive"
        if "ableis" in off:
            offensive[i] = "ableist"
        if "racis" in off:
            offensive[i] = "racist"
        if "stereotype" in off:
            offensive[i] = "offensive stereotype"

    off_binary = (np.array(offensive) != "not offensive").astype(int)
    return off_binary


def output_offensiveness_accuracy(
    df_prediction, reference_data, variable="offensiveness"
):
    offensiveness_prediction = df_prediction["offensiveness"].tolist()
    offensiveness_reference = reference_data[variable]  # type: ignore
    prediction_binary = binarize_offensiveness(offensiveness_prediction)
    reference_binary = offensiveness_reference
    prediction_filtered = []
    reference_filtered = []
    for prediction, label in zip(prediction_binary, reference_binary):
        if label == -1:
            continue
        else:
            prediction_filtered.append(prediction)
            reference_filtered.append(label)
    print(
        f"The total number is {len(prediction_filtered)}, the following is accuracy, recall, precision, f1"
    )
    print(accuracy_score(reference_filtered, prediction_filtered))
    print(recall_score(reference_filtered, prediction_filtered))
    print(precision_score(reference_filtered, prediction_filtered))
    print(f1_score(reference_filtered, prediction_filtered))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file",
        type=str,
        default="./data/inference_data/adversarial_contexts_statements/explanations_v3_original_prompt/mAgr_contexts_explanations.csv",
    )
    parser.add_argument("--reference_file", type=str, default="")
    # parser.add_argument("--variable", type=str, default="offensiveness")
    args = parser.parse_args()

    df_prediction = pd.read_csv(args.prediction_file)
    data_files = {"advtest": "mAgr_contexts_explanations_2.csv"}
    reference_data = datasets.load.load_dataset(
        "context-sbf/context-sbf", split="advtest", data_files=data_files
    )
    output_offensiveness_accuracy(
        df_prediction, reference_data, "offensive_majority"
    )
    output_offensiveness_accuracy(
        df_prediction, reference_data, "offensive_allapp"
    )


if __name__ == "__main__":
    main()
