import numpy as np
import os
import sys
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer 
from frozen_lm_qa.pytorch_model import T5ForPromptTuningQuesionAnswering

# model_of_choice = sys.argv[1]
os.environ['TRANSFORMERS_CACHE'] = '/projects/tir3/users/xuhuiz/socialiq/'
models = {
    'bert-base': "bert-base-cased",
    'bert-large': "bert-large-cased",
    'roberta-base': "roberta-base",
    'roberta-large': "roberta-large",
}
tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
dataset = load_dataset("csv", data_files= {"train": "./data/filters/train.csv", "validation": "./data/filters/valid.csv"})

def tokenize_function_answer(examples):
    return tokenize_function(examples, 'question_answer')

def tokenize_function_full(examples):
    return tokenize_function(examples, 'full')

def tokenize_function(examples, combination):
    examples['labels'] = tokenizer(examples['labels'])['input_ids']
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# tokenized_datasets = dataset.map(tokenize_function_answer, batched=True)
tokenized_datasets = dataset.map(tokenize_function_full, batched=True)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
breakpoint()
model = T5ForPromptTuningQuesionAnswering.from_pretrained("/home/xuhuiz/t5x-frozen-lm-qa-base", from_flax=True)
metric = load_metric("accuracy")

training_args = TrainingArguments(output_dir=f"/projects/tir3/users/xuhuiz/socialiq/test_trainer_frozen_lm_qa", 
                                per_device_train_batch_size=8,
                                per_device_eval_batch_size=16,
                                learning_rate=2e-5,
                                evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# trainer.train()
trainer.predict(tokenized_datasets["validation"])