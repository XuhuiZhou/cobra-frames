# tests/sbf_modeling_code/test_reward_model.py
import os

import numpy as np

from sbf_modeling import RewardModel
from sbf_modeling.utils.data import get_dummy_data

os.environ["WANDB_MODE"] = "offline"


def test_create_reward_model():
    reward_model = RewardModel()
    assert reward_model is not None


def test_reward_model_train_api():
    reward_model = RewardModel()
    model = reward_model.train(get_dummy_data())
    assert isinstance(model, RewardModel)


def test_reward_model_predict_api():
    reward_model = RewardModel()
    dummy_data = get_dummy_data()["validation"]
    predictions = reward_model.predict(dummy_data)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(dummy_data), 7)
