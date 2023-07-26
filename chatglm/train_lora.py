import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_int8_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


def parse_args():
    parser = argparse.ArgumentParser(description="ChatGLM LoRA")
    parser.add_argument("--train_args_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="/data/wuguangshuo/THUDMchatglm-6b/")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--overwrite_cache", type=bool, default=False)
    parser.add_argument("--use_fp16", type=bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default="/home/wuguangshuo/dev_env/chatglm_tuning-main/output/adgen-chatglm-6b-lora/")
    parser.add_argument("--quantization_bit", type=int, default=None)
    parser.add_argument("--model_parallel", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    args = parser.parse_args()
    return args


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        lengths = [len(feature["input_ids"]) for feature in batch]
        longest = max(lengths)
        input_ids, labels = [], []
        for length, feature in sorted(zip(lengths, batch), key=lambda x: -x[0]):
            pad_len = longest - length
            ids = feature["input_ids"] + [self.pad_token_id] * pad_len
            label = feature["labels"] + [-100] * pad_len
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return {"input_ids": input_ids, "labels": labels}


class ModifiedTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train(args):
    parser = HfArgumentParser(TrainingArguments)
    # training_args, = parser.parse_json_file(json_file=args.train_args_file)
    training_args, = parser.parse_json_file(json_file="./chatglm_6b_lora.json")
    # Distributed training
    # if we are in a distributed setting, we need to set the device map and max memory per device
    device_map = "auto"
    if "LOCAL_RANK" in os.environ:
        training_args.local_rank = int(os.environ["LOCAL_RANK"])
        device_map = {"": training_args.local_rank}

    # Set seed
    set_seed(args.seed)
    training_args.seed = args.seed

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True,device_map=device_map)
    #加载量化模型
    if args.quantization_bit is not None:
        print(f"Quantized to {args.quantization_bit} bit")
        model = model.quantize(args.quantization_bit)
    #半精度
    if args.use_fp16 is not None:
        model = model.half()
    #暂时不明白,但是使用梯度累计必须为False,设置为True也会自动修复为False
    # model.config.use_cache = False
    #梯度累计
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    #并行化训练
    if args.model_parallel:
        model.is_parallelizable = True
        model.model_parallel = True

    # Define LoRA Config
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["chatglm"]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)

    #是否加载之前的模型

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        # Full checkpoint
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
            print("加载成功")
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()
    # Load dataset
    data = load_dataset(path="json", data_files=args.data_path)
    column_names = data["train"].column_names

    def tokenize_function(example):
        question = example["instruction"]
        if example.get("input"):
            if example["input"].strip():
                question += f"\n{example['input']}"
        answer = example["output"]

        q_ids = tokenizer.encode(text=question, add_special_tokens=False)
        a_ids = tokenizer.encode(text=answer, add_special_tokens=False)
        if len(q_ids) > args.max_input_length - 1:
            q_ids = q_ids[: args.max_input_length - 1]
        if len(a_ids) > args.max_output_length - 2:
            a_ids = a_ids[: args.max_output_length - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
        question_length = input_ids.index(tokenizer.bos_token_id)
        labels = [-100] * question_length + input_ids[question_length:]
        return {"input_ids": input_ids, "labels": labels}

    train_dataset = data["train"].map(tokenize_function, remove_columns=column_names,load_from_cache_file=args.overwrite_cache)
    eval_dataset = None
    if args.eval_path is not None:
        eval_data = load_dataset(path="json", data_files=args.eval_path)
        eval_dataset = eval_data["train"].map(tokenize_function, remove_columns=column_names)

    # trainer
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(pad_token_id=tokenizer.pad_token_id)
    )

    # train model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save our LoRA model & tokenizer results
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)

