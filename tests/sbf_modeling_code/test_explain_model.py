# tests/sbf_modeling_code/test_explain_model.py
import os

import numpy as np
from transformers import Seq2SeqTrainingArguments

from sbf_modeling import ExplainModel
from sbf_modeling.utils.data import get_dummy_data

os.environ["WANDB_MODE"] = "offline"


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


def test_wandb_log():
    explain_model = ExplainModel()
    os.environ["WANDB_MODE"] = "dryrun"
    model = explain_model.train(
        get_dummy_data(),
        args=Seq2SeqTrainingArguments(
            output_dir="_explain_model",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="steps",
            eval_steps=1,
            logging_steps=100,
            gradient_accumulation_steps=8,
            num_train_epochs=2,
            weight_decay=0.1,
            lr_scheduler_type="cosine",
            learning_rate=1e-4,
            save_steps=5_000,
            generation_max_length=512,
            predict_with_generate=True,  # generation in evaluation
            prediction_loss_only=False,  # generation in evaluation
            report_to=["wandb"],
        ),
    )
    assert isinstance(model, ExplainModel)
