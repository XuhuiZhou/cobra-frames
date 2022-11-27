# tests/sbf_modeling_code/test_reward_model.py
import numpy as np
from datasets.arrow_dataset import Dataset

from sbf_modeling import RewardModel

def get_dummy_data() -> Dataset:
    return Dataset.from_dict(
        dict(
            statement=["the law prohibits the mentally ill from breeding"],
            speakerIdentity=["older man"],
            listenerIdentity=["nobody, just muttered to himself"],
            intent=["The speaker is trying to state a fact"],
            targetGroup=["mentally ill people"],
            relevantPowerDynamics=["power differential between the mentally ill and the general public, as well as the speaker and the mentally ill"],
            implication=["It implies that mentally ill people should not be allowed to have children, and that they are not capable of taking care of them"],
            targetGroupEmotionalReaction=["could be offended or hurt by the statement, might feel like their abilities are being invalidated"],
            targetGroupCognitiveReaction=["might want to avoid places where they could overhear something like that, could make them more cautious of people who think that way"],
            offensiveness=["offensive"],
            labels=[(1,1,1,1,1,1,1)],
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
