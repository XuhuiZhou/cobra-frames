import datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict


def get_dummy_data() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                dict(
                    statement=[
                        "the law prohibits the mentally ill from breeding",
                        "the law prohibits the mentally ill from breeding",
                    ],
                    situationalContext=[
                        "overheard in a grocery store",
                        "overheard in a grocery store",
                    ],
                    speakerIdentity=["older man", "older man"],
                    listenerIdentity=[
                        "nobody, just muttered to himself",
                        "nobody, just muttered to himself",
                    ],
                    intent=[
                        "The speaker is trying to state a fact",
                        "The speaker is trying to state a fact",
                    ],
                    targetGroup=["mentally ill people", "mentally ill people"],
                    relevantPowerDynamics=[
                        "power differential between the mentally ill and the general public, as well as the speaker and the mentally ill",
                        "power differential between the mentally ill and the general public, as well as the speaker and the mentally ill",
                    ],
                    implication=[
                        "It implies that mentally ill people should not be allowed to have children, and that they are not capable of taking care of them",
                        "It implies that mentally ill people should not be allowed to have children, and that they are not capable of taking care of them",
                    ],
                    targetGroupEmotionalReaction=[
                        "could be offended or hurt by the statement, might feel like their abilities are being invalidated",
                        "could be offended or hurt by the statement, might feel like their abilities are being invalidated",
                    ],
                    targetGroupCognitiveReaction=[
                        "might want to avoid places where they could overhear something like that, could make them more cautious of people who think that way",
                        "might want to avoid places where they could overhear something like that, could make them more cautious of people who think that way",
                    ],
                    offensiveness=["offensive", "microagreesion"],
                    labels=[(1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1)],
                )
            ),
            "validation": Dataset.from_dict(
                dict(
                    statement=[
                        "the law prohibits the mentally ill from breeding",
                        "the law prohibits the mentally ill from breeding",
                    ],
                    situationalContext=[
                        "overheard in a grocery store",
                        "overheard in a grocery store",
                    ],
                    speakerIdentity=["older man", "older man"],
                    listenerIdentity=[
                        "nobody, just muttered to himself",
                        "nobody, just muttered to himself",
                    ],
                    intent=[
                        "The speaker is trying to state a fact",
                        "The speaker is trying to state a fact",
                    ],
                    targetGroup=["mentally ill people", "mentally ill people"],
                    relevantPowerDynamics=[
                        "power differential between the mentally ill and the general public, as well as the speaker and the mentally ill",
                        "power differential between the mentally ill and the general public, as well as the speaker and the mentally ill",
                    ],
                    implication=[
                        "It implies that mentally ill people should not be allowed to have children, and that they are not capable of taking care of them",
                        "It implies that mentally ill people should not be allowed to have children, and that they are not capable of taking care of them",
                    ],
                    targetGroupEmotionalReaction=[
                        "could be offended or hurt by the statement, might feel like their abilities are being invalidated",
                        "could be offended or hurt by the statement, might feel like their abilities are being invalidated",
                    ],
                    targetGroupCognitiveReaction=[
                        "might want to avoid places where they could overhear something like that, could make them more cautious of people who think that way",
                        "might want to avoid places where they could overhear something like that, could make them more cautious of people who think that way",
                    ],
                    offensiveness=["offensive", "microagreesion"],
                    labels=[(1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1)],
                )
            ),
        }
    )


def get_data(mode: str, split: str = ""):
    if mode == "deployment":
        return get_train_data(split)
    elif mode == "tests":
        return get_dummy_data() if split == "" else get_dummy_data()[split]
    else:
        raise ValueError(f"Unknown mode {mode}")


def get_train_data(split: str = "") -> DatasetDict:
    if split == "":
        return datasets.load.load_dataset("context-sbf/context-sbf")  # type: ignore # we know this is a Dataset
    elif split == "adv":
        data_files = {"advtest": "mAgr_contexts_explanations_2.csv"}
        return datasets.load.load_dataset("context-sbf/context-sbf", split="advtest", data_files=data_files)  # type: ignore
    else:
        return datasets.load.load_dataset("context-sbf/context-sbf", split=split)  # type: ignore
