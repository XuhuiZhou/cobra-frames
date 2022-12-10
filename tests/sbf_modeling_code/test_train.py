import os

from sbf_modeling import ExplainModel, train
from sbf_modeling.utils.data import get_dummy_data

os.environ["WANDB_MODE"] = "offline"


def test_train_func():
    explain_model = ExplainModel()
    model = explain_model.train(get_dummy_data())
    model_dir = ".log/_explain_model"
    train.train(model=model, train_data=get_dummy_data(), model_dir=model_dir)


def test_train_script():
    import os

    assert (
        os.system(
            'python sbf_modeling/train.py --gin_file="scripts/explain_model/explain_model.gin" --gin.MODEL_DIR="\'.log/explain-model\'"'
        )
        == 0
    )
