from __future__ import annotations

import re
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from datasets.arrow_dataset import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

from sbf_modeling import BaseSBFModel
from sbf_modeling.prompt_templates import (
    map_dataset_to_prompt_prefix,
    map_dataset_to_tokenized_prompt,
)


class ExplainModel(BaseSBFModel):
    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-small",
    ):
        assert "t5" in t5_model_name, "Reward model only supports T5 models."
        try:
            self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(t5_model_name)  # type: ignore
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        except Exception as e:
            print(f"Error loading model {t5_model_name}: {e}")
            raise e

    def train(
        self,
        dataset: Dataset,
        args: TrainingArguments = TrainingArguments(
            output_dir="explain-model",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            eval_steps=5_000,
            logging_steps=5_000,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            weight_decay=0.1,
            warmup_steps=1_000,
            lr_scheduler_type="cosine",
            learning_rate=5e-4,
            save_steps=5_000,
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
        prompt_dataset = dataset.map(
            partial(map_dataset_to_tokenized_prompt, self.tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(
            self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=prompt_dataset,  # type: ignore
            eval_dataset=prompt_dataset,  # type: ignore # TODO: use a separate eval dataset
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
        pipe = pipeline(
            "text-generation",
            model="explain-model",
        )

        prefix_datset = dataset.map(
            map_dataset_to_prompt_prefix,
            batched=True,
            remove_columns=dataset.column_names,
        )
        prefixes = prefix_datset["prefix"]

        generated_text: List[str] = list(
            map(
                lambda i: i[0]["generated_text"],
                pipe(prefixes, num_return_sequences=1),
            )
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
