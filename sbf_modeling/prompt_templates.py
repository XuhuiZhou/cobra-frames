from typing import Dict, List, cast

import gin
import numpy as np
from datasets.arrow_dataset import Dataset
from transformers import PreTrainedTokenizer

CONTEXT_LENGTH = 512
TARGET_LENGTH = 512


@gin.configurable
def create_context_template(
    with_context: bool = True,
):
    if with_context:
        return "This is {situationalContext}, {speakerIdentity} speaking to {listenerIdentity}: {statement}\n"
    else:
        return "Please analyze the intent, target group, power dynamics, implication, the emotional and cognitive reaction of the target group and offensiveness of {statement}\n"


CONTEXT_TEMPLATE = create_context_template()

QUESTION_TEMPLATES = dict(
    intent="What is the intent of the speaker?",
    targetGroup="Who are the statement targeting at?",
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
    context_input = map(
        lambda instance: CONTEXT_TEMPLATE.format(**instance),
        (dict(zip(element_cast, t)) for t in zip(*element_cast.values())),
    )
    target = map(
        lambda instance: "\n".join(
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
        ),
        (dict(zip(element_cast, t)) for t in zip(*element_cast.values())),
    )
    tokenized_context_input = tokenizer(
        list(context_input),
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_overflowing_tokens=True,
        return_length=True,
    )
    tokenized_target = tokenizer(
        list(target),
        truncation=True,
        max_length=TARGET_LENGTH,
        return_overflowing_tokens=True,
        return_length=True,
    )
    tokenized_context_input.update({"labels": tokenized_target["input_ids"]})
    return tokenized_context_input  # type: ignore
