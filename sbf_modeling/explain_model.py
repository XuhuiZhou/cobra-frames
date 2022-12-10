from __future__ import annotations

import os
import re
from functools import partial
from typing import Dict, List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import tqdm
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    DataCollatorForSeq2Seq,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers.integrations import WandbCallback

from sbf_modeling import BaseSBFModel
from sbf_modeling.metrics import (
    aggregated_metrics,
    aggregated_metrics_with_postprocess,
    bleu_metrics,
    prediction_metrics,
)
from sbf_modeling.prompt_templates import (
    map_dataset_to_tokenized_prompt,
)

os.environ["WANDB_PROJECT"] = "context-sbf"


class ExplainModel(BaseSBFModel):
    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-small",
        from_local: bool = False,
    ):
        assert (
            "t5" in t5_model_name or from_local
        ), "Reward model only supports T5 models."
        try:
            self.model: T5ForConditionalGeneration = cast(
                T5ForConditionalGeneration,
                T5ForConditionalGeneration.from_pretrained(t5_model_name),
            )
        except Exception as e:
            print(f"Error loading model {t5_model_name}: {e}")
            raise e

        self.model: T5ForConditionalGeneration = cast(  # type: ignore
            T5ForConditionalGeneration, self.model
        )
        if t5_model_name in ["google/flan-t5-xxl", "google/flan-t5-xl"]:
            self.model.parallelize()
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            t5_model_name
        )

    @classmethod
    def from_pretrained(cls, model_dir: str) -> ExplainModel:
        model = cls(model_dir, from_local=True)
        return model

    def train(
        self,
        dataset: DatasetDict,
        args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
            output_dir=".log/_explain_model",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="steps",
            eval_steps=1,
            logging_steps=1,
            gradient_accumulation_steps=1,
            num_train_epochs=2,
            weight_decay=0.1,
            lr_scheduler_type="cosine",
            learning_rate=1e-4,
            save_steps=5_000,
            generation_max_length=512,
            predict_with_generate=True,  # generation in evaluation
            prediction_loss_only=False,  # generation in evaluation
        ),
        save_model_dir: str = "explain-model",
    ) -> ExplainModel:
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
        dataset["validation"] = Dataset.from_dict(dataset["validation"][:300])
        prompt_train_dataset = dataset["train"].map(
            partial(map_dataset_to_tokenized_prompt, self.tokenizer),
            batched=True,
            load_from_cache_file=False,
            remove_columns=dataset["train"].column_names,
        )
        prompt_train_dataset = cast(TorchDataset, prompt_train_dataset)  # type: ignore
        prompt_valid_dataset = dataset["validation"].map(
            partial(map_dataset_to_tokenized_prompt, self.tokenizer),
            batched=True,
            load_from_cache_file=False,
            remove_columns=dataset["validation"].column_names,
        )
        prompt_valid_dataset = cast(TorchDataset, prompt_valid_dataset)  # type: ignore

        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)

        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=prompt_train_dataset,
            eval_dataset=prompt_valid_dataset,
            compute_metrics=partial(
                aggregated_metrics_with_postprocess,
                [partial(prediction_metrics, 300), bleu_metrics],
                self.tokenizer,
            ),
        )
        trainer.train()

        if save_model_dir != "":
            trainer.save_model(save_model_dir)

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
        prompt_dataset = dataset.map(
            partial(map_dataset_to_tokenized_prompt, self.tokenizer),
            batched=True,
            load_from_cache_file=False,
            remove_columns=dataset.column_names,
        )
        tokenized_prompt = prompt_dataset["input_ids"]

        generated_text: List[str] = list(
            self.tokenizer.batch_decode(
                self.model.generate(
                    input_ids=torch.Tensor(input_ids)
                    .unsqueeze(0)
                    .long()
                    .to(self.model.device),
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                )
            )[0]
            for input_ids in tqdm.tqdm(tokenized_prompt)
        )

        keys = [
            "intent",
            "targetGroup",
            "relevantPowerDynamics",
            "implication",
            "targetGroupEmotionalReaction",
            "targetGroupCognitiveReaction",
            "offensiveness",
        ]
        answer_dict: Dict[str, List[str]] = {key: [] for key in keys}
        for txt in generated_text:
            answers = re.findall(r"A: (.*?)\n", txt)
            if len(answers) < 7:
                answers += [""] * (7 - len(answers))
            answers = answers[:7]
            for key, answer in zip(keys, answers):
                answer_dict[key].append(answer)

        return answer_dict
