# tests/sbf_modeling_code/test_reward_model.py
import numpy as np
from datasets.arrow_dataset import Dataset

from sbf_modeling import RewardModel


def get_dummy_data() -> Dataset:
    return Dataset.from_dict(
        dict(
            intent=[],
            targetGroup=[],
            relevantPowerDynamics=[],
            implication=[],
            targetGroupEmotionalReaction=[],
            targetGroupCognitiveReaction=[],
            offensiveness=[],
            labels=[],
        )
    )


def test_create_reward_model():
    reward_model = RewardModel()
    assert reward_model is not None


def test_reward_model_train_api():
    reward_model = RewardModel()
    model = reward_model.train(get_dummy_data())
    assert isinstance(model, RewardModel)


def test_reward_model_predict_api():
    reward_model = RewardModel()
    dummy_data = get_dummy_data()
    predictions = reward_model.predict(dummy_data)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(dummy_data), 7)
