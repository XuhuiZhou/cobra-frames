from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from datasets.arrow_dataset import Dataset
from transformers import T5ForConditionalGeneration


class ExplainModel(object):
    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-small",
    ):
        assert "t5" in t5_model_name, "Reward model only supports T5 models."
        try:
            self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(t5_model_name)  # type: ignore
        except Exception as e:
            print(f"Error loading model {t5_model_name}: {e}")
            raise e

    def train(self, dataset: Dataset) -> ExplainModel:
        """
        Train the reward model on the given dataset.

        Args:
            dataset (Dataset): The dataset to train on. A single example is a dictionary with the following keys:
                - statement (str): The statement to be evaluated.
                - situationalContext (str): The context of the situation.
                - speakerIdentity (str): The identity of the speaker.
                - listenerIdentity (str): The identity of the listener.
                - intent (str): The intent of the statement.
                - targetGroup (str): The target group of the statement.
                - relevantPowerDynamics (str): The relevant power dynamics of the statement.
                - implication (str): The implication of the statement.
                - targetGroupEmotionalReaction (str): The emotional reaction of the target group to the statement.
                - targetGroupCognitiveReaction (str): The cognitive reaction of the target group to the statement.
                - offensiveness (str): The offensiveness of the statement.

        Returns:
            RewardModel: The trained reward model.
        """
        return self

    def predict(self, dataset: Dataset) -> Dict[str, List[str]]:
        """
        Predict the reward for the given dataset.

        Args:
            dataset (Dataset): The dataset to predict on. A single example is a dictionary with the same keys as train above except for "labels".

        Returns:
            Dict[str, List[str]]: The predicted explanation for each example in the dataset, with the following keys:
                -
        """

        return dict(
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
        )
