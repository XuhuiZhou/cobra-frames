import datasets

from tests.sbf_modeling_code.utils import get_dummy_data


def get_data(mode: str):
    if mode == "deployment":
        return get_train_data()
    elif mode == "tests":
        return get_dummy_data()
    else:
        raise ValueError(f"Unknown mode {mode}")


def get_train_data() -> datasets.dataset_dict.DatasetDict:
    return datasets.load.load_dataset("context-sbf/context-sbf")  # type: ignore # we know this is a Dataset
