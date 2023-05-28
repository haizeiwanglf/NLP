import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    set_seed
from torch.utils.data import Dataset
from transformers.trainer_utils import IntervalStrategy
import torch
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig
)

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import time

set_seed(42)
model_name = "bert-base-chinese"
num_labels = 10

if any(k in model_name for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"
model_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)
# 训练器配置
p_type = "lora"
if p_type == "prefix-tuning":
    peft_type = PeftType.PREFIX_TUNING
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
elif p_type == "prompt-tuning":
    peft_type = PeftType.PROMPT_TUNING
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
elif p_type == "p-tuning":
    peft_type = PeftType.P_TUNING
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
elif p_type == "lora":
    peft_type = PeftType.LORA
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)

if p_type is not None:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


def load_data(path):
    with open(path, 'r', encoding='UTF-8') as f:
        data = []
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            data.append((content, int(label)))
    random.shuffle(data)
    return data

class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.texts = [line[0] for line in data]
        self.labels = [line[1] for line in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        return [text, label]

def collate_fn(features):
    text = [line[0] for line in features]
    label = [line[1] for line in features]
    tokenized = tokenizer(text, padding=True, return_tensors='pt')
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    labels = torch.tensor(label)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_data = load_data("./data/train.txt")
dev_data = load_data("./data/dev.txt")
test_data = load_data("./data/test.txt")

train = MyDataset(train_data)
valid = MyDataset(dev_data)
test = MyDataset(test_data)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds.flatten(), average='macro')
    print({'accuracy': round((labels == preds).mean(), 6),
           'f1': round(f1, 6),
           'precision': round(precision, 6),
           'recall': round(recall, 6)})
    return {'accuracy': round((labels == preds).mean(), 6),
            'f1': round(f1, 6),
            'precision': round(precision, 6),
            'recall': round(recall, 6)}

training_args=TrainingArguments(output_dir='./weights',
                                    do_train=True,
                                    do_eval=True,
                                    do_predict=True,
                                    per_device_train_batch_size=64,
                                    per_device_eval_batch_size=64,
                                    eval_steps=400,
                                    logging_steps=100,
                                    report_to=["wandb"],
                                    save_steps=400,
                                    metric_for_best_model='f1',
                                    logging_strategy=IntervalStrategy.STEPS,
                                    evaluation_strategy=IntervalStrategy.STEPS,
                                    save_strategy=IntervalStrategy.STEPS,
                                    save_total_limit=1
                               )

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train,
                  eval_dataset=valid,
                  data_collator=collate_fn,
                  compute_metrics=compute_metrics
                  )

start = time.time()
trainer.train()
model.save_pretrained(p_type)
end = time.time()
print("训练耗时：{}分钟".format((end - start) / 60))

res=trainer.predict(test)
print(res.metrics)
print("finish")

#inference
# from peft import PeftModel
# import torch
# from transformers import AutoModelForSequenceClassification,AutoTokenizer
# model_name = "bert-base-chinese"
# lora_model_dir = "./lora"
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# lora_model = PeftModel.from_pretrained(model, lora_model_dir)
# lora_model.cuda()
# lora_model.eval()
# text = '中国人民公安大学2012年硕士研究生目录及书目'
# with torch.no_grad():
#   inputs = tokenizer(text, truncation=True, max_length=86, padding="max_length", return_tensors="pt")
#   for k, v in inputs.items():
#     inputs[k] = v.cuda()
#   output = lora_model(**inputs)
#   logits = output.logits
#   print(logits)