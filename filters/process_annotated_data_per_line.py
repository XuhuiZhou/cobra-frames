import argparse
import os

import numpy as np
import pandas as pd
import sklearn

np.random.seed(12)


def map(number):
    if number == 1:
        return "yes"
    else:
        return "no"


def process_type_data(data):
    inputs = []
    outputs = []
    for index, row in data.iterrows():
        inputs.append(
            "[Statement] "
            + row["Input.statement"].strip()
            + " [Situational context] "
            + row["Input.speechContext"]
            + " [Speaker identity] "
            + row["Input.speakerIdentity"]
            + " [Listener identity] "
            + row["Input.listenerIdentity"]
            + " [Does the situational context supply appropriate and relevant information to set the scene and help you better understand the statement?] \n"
        )
        outputs.append(map(row["Answer.hsituationRating"]) + "\n")

        inputs.append(
            "[Statement] "
            + row["Input.statement"].strip()
            + " [Situational context] "
            + row["Input.speechContext"]
            + " [Speaker identity] "
            + row["Input.speakerIdentity"]
            + " [Listener identity] "
            + row["Input.listenerIdentity"]
            + " [Does the situational context seem plausible/realistic?] \n"
        )
        outputs.append(map(row["Answer.psituationRating"]) + "\n")

        inputs.append(
            "[Statement] "
            + row["Input.statement"].strip()
            + " [Situational context] "
            + row["Input.speechContext"]
            + " [Speaker identity] "
            + row["Input.speakerIdentity"]
            + " [Listener identity] "
            + row["Input.listenerIdentity"]
            + " [Does the speaker identity seem plausible/realistic?] \n"
        )
        outputs.append(map(row["Answer.speakerIdenRating"]) + "\n")

        inputs.append(
            "[Statement] "
            + row["Input.statement"].strip()
            + " [Situational context] "
            + row["Input.speechContext"]
            + " [Speaker identity] "
            + row["Input.speakerIdentity"]
            + " [Listener identity] "
            + row["Input.listenerIdentity"]
            + " [Does the listener identity seem plausible/realistic?] \n"
        )
        outputs.append(map(row["Answer.listenerIdenRating"]) + "\n")

    return inputs, outputs


def combine_to_csv(split, source, target):
    data = {"text": source, "labels": target}
    df = pd.DataFrame.from_dict(data)
    df.to_csv(split, index=False)
    print("Output csv done")


def split_data(args):
    data = pd.read_csv(args.input_data)

    train_0, valid_0, test_0 = np.split(
        data.loc[data["Answer.finalRating"] == 0],
        [
            int(0.8 * len(data.loc[data["Answer.finalRating"] == 0])),
            int(0.9 * len(data.loc[data["Answer.finalRating"] == 0])),
        ],
    )
    train_1, valid_1, test_1 = np.split(
        data.loc[data["Answer.finalRating"] == 1],
        [
            int(0.8 * len(data.loc[data["Answer.finalRating"] == 1])),
            int(0.9 * len(data.loc[data["Answer.finalRating"] == 1])),
        ],
    )

    print(
        "Final false - Train:{} Valid:{} Test: {}".format(
            len(train_0), len(valid_0), len(test_0)
        )
    )
    print(
        "Final true - Train:{} Valid:{} Test: {}".format(
            len(train_1), len(valid_1), len(test_1)
        )
    )

    train = pd.concat([pd.DataFrame(train_0), pd.DataFrame(train_1)])
    valid = pd.concat([pd.DataFrame(valid_0), pd.DataFrame(valid_1)])
    test = pd.concat([pd.DataFrame(test_0), pd.DataFrame(test_1)])
    train = sklearn.utils.shuffle(train)
    valid = sklearn.utils.shuffle(valid)
    test = sklearn.utils.shuffle(test)

    inputs_train, outputs_train = process_type_data(train)

    inputs_valid, outputs_valid = process_type_data(valid)

    inputs_test, outputs_test = process_type_data(test)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "train.source"), "w") as f:
        print(len(inputs_train))
        f.writelines(inputs_train)

    with open(os.path.join(args.output_dir, "train.target"), "w") as f:
        print(len(outputs_train))
        f.writelines(outputs_train)

    with open(os.path.join(args.output_dir, "valid.source"), "w") as f:
        print(len(inputs_valid))
        f.writelines(inputs_valid)

    with open(os.path.join(args.output_dir, "valid.target"), "w") as f:
        print(len(outputs_valid))
        f.writelines(outputs_valid)

    with open(os.path.join(args.output_dir, "test.source"), "w") as f:
        print(len(inputs_test))
        f.writelines(inputs_test)

    with open(os.path.join(args.output_dir, "test.target"), "w") as f:
        print(len(outputs_test))
        f.writelines(outputs_test)

    combine_to_csv(
        os.path.join(args.output_dir, "train.csv"), inputs_train, outputs_train
    )
    combine_to_csv(
        os.path.join(args.output_dir, "valid.csv"), inputs_valid, outputs_valid
    )
    combine_to_csv(
        os.path.join(args.output_dir, "test.csv"), inputs_test, outputs_test
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data",
        type=str,
        default=f'{os.environ["HOME"]}/projects/xuhuiz/context-sbf/data/filters/chunk2-annotation_summary_acc.csv',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f'{os.environ["HOME"]}/projects/xuhuiz/context-sbf/data/filters/',
    )
    args = parser.parse_args()

    split_data(args)
