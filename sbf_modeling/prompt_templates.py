from typing import Dict, List, cast

import numpy as np
from datasets.arrow_dataset import Dataset
from transformers import PreTrainedTokenizer

CONTEXT_LENGTH = 512

CONTEXT_TEMPLATE = "This is {situationalContext}, {speakerIdentity} speaking to {listenerIdentity}: {statement}\n"

QUESTION_TEMPLATES = dict(
    intent="What is the intent of the speaker?",
    targetGroup="Who are the speaker targeting at?",
    relevantPowerDynamics="What are the relevant power dynamics?",
    implication="What is the implication of the statement?",
    targetGroupEmotionalReaction="What is the emotional reaction of the target group?",
    targetGroupCognitiveReaction="What is the cognitive reaction of the target group?",
    offensiveness="How offensive is the statement?",
)


def map_dataset_to_tokenized_prompt(
    tokenizer: PreTrainedTokenizer, element: Dataset
) -> Dict[str, np.ndarray]:
    element_cast: Dict[str, List[str]] = cast(Dict[str, List[str]], element)  # type: ignore
    whole_prompt = map(
        lambda instance: (
            CONTEXT_TEMPLATE.format(**instance)
            + "\n".join(
                f"Q: {QUESTION_TEMPLATES[key]} A: {instance[key]}"
                for key in [
                    "intent",
                    "targetGroup",
                    "relevantPowerDynamics",
                    "implication",
                    "targetGroupEmotionalReaction",
                    "targetGroupCognitiveReaction",
                    "offensiveness",
                ]
            )
        )
        + "\n",
        (dict(zip(element_cast, t)) for t in zip(*element_cast.values())),
    )
    outputs = tokenizer(
        list(whole_prompt),
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_overflowing_tokens=True,
        return_length=True,
        return_tensors="np",
    )
    return {"input_ids": outputs["input_ids"]}  # type: ignore


def map_dataset_to_prompt_prefix(element: Dataset) -> Dict[str, List[str]]:
    element_cast: Dict[str, List[str]] = cast(Dict[str, List[str]], element)  # type: ignore
    whole_prompt = map(
        lambda instance: (CONTEXT_TEMPLATE.format(**instance)),
        (dict(zip(element_cast, t)) for t in zip(*element_cast.values())),
    )

    return {"prefix": list(whole_prompt)}
