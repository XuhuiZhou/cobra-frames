from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict


def get_dummy_data() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                dict(
                    statement=[
                        "the law prohibits the mentally ill from breeding"
                    ],
                    situationalContext=["overheard in a grocery store"],
                    speakerIdentity=["older man"],
                    listenerIdentity=["nobody, just muttered to himself"],
                    intent=["The speaker is trying to state a fact"],
                    targetGroup=["mentally ill people"],
                    relevantPowerDynamics=[
                        "power differential between the mentally ill and the general public, as well as the speaker and the mentally ill"
                    ],
                    implication=[
                        "It implies that mentally ill people should not be allowed to have children, and that they are not capable of taking care of them"
                    ],
                    targetGroupEmotionalReaction=[
                        "could be offended or hurt by the statement, might feel like their abilities are being invalidated"
                    ],
                    targetGroupCognitiveReaction=[
                        "might want to avoid places where they could overhear something like that, could make them more cautious of people who think that way"
                    ],
                    offensiveness=["offensive"],
                    labels=[(1, 1, 1, 1, 1, 1, 1)],
                )
            ),
            "validation": Dataset.from_dict(
                dict(
                    statement=[
                        "the law prohibits the mentally ill from breeding"
                    ],
                    situationalContext=["overheard in a grocery store"],
                    speakerIdentity=["older man"],
                    listenerIdentity=["nobody, just muttered to himself"],
                    intent=["The speaker is trying to state a fact"],
                    targetGroup=["mentally ill people"],
                    relevantPowerDynamics=[
                        "power differential between the mentally ill and the general public, as well as the speaker and the mentally ill"
                    ],
                    implication=[
                        "It implies that mentally ill people should not be allowed to have children, and that they are not capable of taking care of them"
                    ],
                    targetGroupEmotionalReaction=[
                        "could be offended or hurt by the statement, might feel like their abilities are being invalidated"
                    ],
                    targetGroupCognitiveReaction=[
                        "might want to avoid places where they could overhear something like that, could make them more cautious of people who think that way"
                    ],
                    offensiveness=["offensive"],
                    labels=[(1, 1, 1, 1, 1, 1, 1)],
                )
            ),
        }
    )
