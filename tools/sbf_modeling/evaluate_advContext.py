import argparse

import datasets
import numpy as np
import pandas as pd


def calculate_accuracy(predictions, labels):
    return (predictions == labels).astype(int).mean()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file",
        type=str,
        default=".log/explain-model-xl/adv/answer.csv",
    )
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--variable", type=str, default="offensiveness")
    args = parser.parse_args()

    df_prediction = pd.read_csv(args.prediction_file)
    data_files = {"advtest": "mAgr_contexts_explanations.csv"}
    reference_data = datasets.load.load_dataset(
        "context-sbf/context-sbf", split="advtest", data_files=data_files
    )
    offensiveness_prediction = df_prediction[args.variable].tolist()
    offensiveness_reference = reference_data[args.variable]  # type: ignore
    preset_offensiveness = reference_data["moreOff"]  # type: ignore
    prediction_binary = binarize_offensiveness(offensiveness_prediction)
    reference_binary = binarize_offensiveness(offensiveness_reference)
    preset_binary = np.array(
        [0 if x == -1 else 1 for x in preset_offensiveness]
    )

    print(calculate_accuracy(prediction_binary, reference_binary))
    print(calculate_accuracy(prediction_binary, preset_binary))
    print(calculate_accuracy(reference_binary, preset_binary))


if __name__ == "__main__":
    main()
