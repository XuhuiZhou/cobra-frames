# tests/sbf_modeling_code/test_explain_model.py
import numpy as np
from utils import get_dummy_data

from sbf_modeling import ExplainModel


def test_create_explain_model():
    explain_model = ExplainModel()
    assert explain_model is not None


def test_explain_model_train_api():
    explain_model = ExplainModel()
    model = explain_model.train(get_dummy_data())
    assert isinstance(model, ExplainModel)


def test_explain_model_predict_api():
    explain_model = ExplainModel()
    dummy_data = get_dummy_data()["validation"]
    predictions = explain_model.predict(dummy_data)
    assert isinstance(predictions, dict)
    for key in [
        "intent",
        "targetGroup",
        "relevantPowerDynamics",
        "implication",
        "targetGroupEmotionalReaction",
        "targetGroupCognitiveReaction",
        "offensiveness",
    ]:
        assert key in predictions
        assert isinstance(predictions[key], list)
        assert len(predictions[key]) == len(dummy_data)
        assert all(isinstance(p, str) for p in predictions[key])
