import datasets

from tests.sbf_modeling_code.utils import get_dummy_data


def get_data(mode: str, split: str = ""):
    if mode == "deployment":
        return get_train_data(split)
    elif mode == "tests":
        return get_dummy_data()
    else:
        raise ValueError(f"Unknown mode {mode}")


def get_train_data(split: str = "") -> datasets.dataset_dict.DatasetDict:
    if split == "":
        return datasets.load.load_dataset("context-sbf/context-sbf")  # type: ignore # we know this is a Dataset
    else:
        return datasets.load.load_dataset("context-sbf/context-sbf", split=split)  # type: ignore
