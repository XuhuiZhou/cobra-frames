import os
import argparse
import random
import numpy as np
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pytorch_lightning as pl
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors='pt'):
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for text in f.readlines():
            tokenized = tokenizer.batch_encode_plus(
                [text + ' </s>'], truncation=True, max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
    return examples


class CustomDataset(Dataset):
    def __init__(self, tokenizer, 
        data_dir='data_single/', 
        type_path='train', 
        max_length = 180,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.source_ids = []
        self.source_mask = []
        self.labels = []
        data_path_source = os.path.join(data_dir, type_path + ".source")
        data_path_target = os.path.join(data_dir, type_path + ".target")

        self.source = encode_file(self.tokenizer, data_path_source, max_length)
        self.target = encode_file(self.tokenizer, data_path_target, max_length)


    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        # return self.source_ids[index], self.source_mask[index], self.labels[index]
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        target_mask = self.target[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

def get_dataset(tokenizer, data_dir, type_path):
    return CustomDataset(tokenizer = tokenizer, data_dir = data_dir, type_path=type_path)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        
        self.model = AutoModelWithLMHead.from_pretrained(hparams.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
  
    def is_logger(self):
        return True
  
    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
  
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        outputs = {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
        print("\n avg_train_loss: {} \n".format(avg_train_loss))

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.hparams.data_dir, type_path="train")
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.hparams.data_dir, type_path="valid")
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)



def evaluate(loader, model):
    outputs = []
    targets = []
    for batch in tqdm(loader):
        outs = model.model.generate(input_ids=batch['source_ids'], 
                                attention_mask=batch['source_mask'], 
                                max_length=2)

        dec = [int(model.tokenizer.decode(ids).replace('<pad> ', '')) for ids in outs]
        target = [int(model.tokenizer.decode(ids).split('</s>')[0]) for ids in batch["target_ids"]]
        
        outputs.extend(dec)
        targets.extend(target)

    return outputs, targets

def calculate_scores(output, target):
    acc = (output == target).sum() / len(target)
    precision = (output * target).sum() / output.sum()
    recall = (output * target).sum() / target.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallelize',type=bool,default=False)
    parser.add_argument('--use_pretrain',type=bool,default=True)
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--data_dir', type=str, default='data_single/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--lr_warmup', type=float, default=0.002)
   
    parser.add_argument('--grad_accum_steps', type=int, default = 16)
    parser.add_argument('--warmup_steps', type=int, default = 0)
    parser.add_argument('--model_type', type=str, default = "t5-base")
    args = parser.parse_args()
    print(args)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #args.n_gpu = torch.cuda.device_count()
    args.n_gpu = 1
    print("device", device, "n_gpu", args.n_gpu)

    batch_size = args.n_batch
    epochs = args.n_iter
    gradient_accumulation_steps = args.grad_accum_steps
    warmup_steps = args.warmup_steps
    model_type= args.model_type
    save_dir = os.path.join(args.model_dir, model_type)

    input_tokens = ['[Statement]', '[group]', '[speech_context]', '[speaker_identity]', '[listener_identity]']
    types_tokens = ['[hSituationalRating]', '[pSituationalRating]', '[speakerIdenRating]', '[listenerIdenRating]']
    
    args_dict = dict(
        data_dir = args.data_dir,
        model_name=model_type,
        learning_rate=args.lr,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=warmup_steps,
        train_batch_size=args.n_batch,
        eval_batch_size=args.n_batch,
        num_train_epochs=args.n_iter,
        gradient_accumulation_steps=gradient_accumulation_steps,
        n_gpu=args.n_gpu,
        early_stop_callback=False,
        opt_level='O1',
        max_grad_norm=0.5,
        seed=args.seed)
    
    pargs = argparse.Namespace(**args_dict)
    model = T5FineTuner(pargs)


    train_params = dict(
        accumulate_grad_batches=16,
        gpus=args.n_gpu,
        max_epochs=epochs,
        precision= 16,
        gradient_clip_val=0.5,
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    torch.save(model.model.state_dict(), args.model_dir + 't5-small/final.pt')

    valid_dataset = CustomDataset(model.tokenizer, data_dir=args.data_dir, type_path='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    val_outputs, val_targets = evaluate(valid_dataloader, model)
    print(calculate_scores(np.array(val_outputs), np.array(val_targets)))

    test_dataset = CustomDataset(model.tokenizer, data_dir=args.data_dir, type_path='test')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    test_outputs, test_targets = evaluate(test_dataloader, model)
    print(calculate_scores(np.array(test_outputs), np.array(test_targets)))
    