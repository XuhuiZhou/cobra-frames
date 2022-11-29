from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from datasets.arrow_dataset import Dataset
from transformers import T5ForConditionalGeneration

from sbf_modeling import BaseSBFModel


class RewardModel(BaseSBFModel):
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

    def train(self, dataset: Dataset) -> RewardModel:
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
                - labels (Tuple[int, int, int, int, int, int, int]): The labels for the example.

        Returns:
            RewardModel: The trained reward model.
        """
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the reward for the given dataset.

        Args:
            dataset (Dataset): The dataset to predict on. A single example is a dictionary with the same keys as train above except for "labels".

        Returns:
            np.ndarray: The predicted reward for each example in the dataset.
        """
        return np.zeros((len(dataset), 7))
