# chatglm_tuning: 基于 ChatGLM-6B的三种 高效参数微调

[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 是一个清华开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。

在医疗数据集中，本仓库实现了对于 ChatGLM-6B 模型基于 LoRA ，freeze模型后几层和 P-Tuning v2 的参数高效微调。

数据集下载地址:[FreedomIntelligence/huatuo_knowledge_graph_qa · Datasets at Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/huatuo_knowledge_graph_qa)

## Requirements

- transformers==4.30


## P-Tuning v2

[P-Tuning v2](https://github.com/THUDM/P-tuning-v2) 是清华大学开源的语言模型提示微调方法，在 3 亿到 100 亿参数的广阔参数规模上，均能仅以 0.1%～3% 的微调参数量，取得和精调方法媲美的迁移效果。

下面以huatuo医疗模型的数据集为实例进行微调

```json
{
    "instruction": "颜面部凹陷的手术治疗有些什么？", 
    "output": "自体颗粒脂肪移植；自体脂肪移植；自体脂肪干细胞移植；自体脂肪颗粒移植"
}
```

对原始数据进行预处理，改变其列名:

```shell
cd dataset/FreedomIntelligencehuatuo_knowledge_graph_qa/
python convert_to_instruct.py
```

运行以下命令进行训练：

```shell
export CUDA_VISIBLE_DEVICES = 0  train_ptuning.py \
--train_args_file ./config/chatglm_6b_ptuning.json \
--data_path ./dataset/FreedomIntelligencehuatuo_knowledge_graph_qa/train_huatuo.json \
--pre_seq_len 128 \

```

如果想进行单机多卡分布式训练，可运行如下命令：

```shell
export CUDA_VISIBLE_DEVICES = 0, 1, 2, 3
torchrun --nproc_per_node=4  train_ptuning.py \
--train_args_file ./config/chatglm_6b_ptuning.json \
--data_path ./dataset/FreedomIntelligencehuatuo_knowledge_graph_qa/train_huatuo.json \
--pre_seq_len 128 \

```

chatglm_6b_ptuning.json 为训练参数配置文件：

```json
{
    "output_dir": "output/adgen-chatglm-6b-ptuning",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-2,
    "num_train_epochs": 1.0,
    "max_steps": 3000,
    "lr_scheduler_type": "cosine",
    "logging_steps": 5,
    "save_strategy": "no",
    "optim": "adamw_torch",
    "fp16": false,
    "remove_unused_columns": false,
    "ddp_find_unused_parameters": false,
    "report_to": "tensorboard"
}
```

在json中，max_steps与num_train_epochs参数分别控制最大步数和训练epoch数目，当两者冲突时优先选择前者，如果num_train_epochs为1，那么训练的步数为 num_sample/(per_device_train_batch_size*device_num+gradient_accumulation_steps)

在 P-tuning v2 训练时模型只保存 PrefixEncoder 部分的参数，所以在推理时需要同时加载原 ChatGLM-6B 模型以及 PrefixEncoder 的权重。



## LoRA

[LoRA](https://github.com/microsoft/LoRA) 的实现思想很简单，如下图所示，就是冻结一个预训练语言模型的矩阵参数，并选择用 `A` 和 `B` 矩阵来替代，在下游任务时只更新 `A` 和 `B`。

![](E:\chatglm_tuning-main\images\lora.png)

 LoRA 的实现流程概况如下：

- 在原始预训练语言模型 (PLM) 旁增加一个旁路，做一个先降维再升维的操作，以此来模拟所谓的内在秩；

- 训练的时候固定 PLM 的参数不变，只训练降维矩阵 `A` 和升维矩阵 `B`，即优化器只优化右路的参数；

- 模型的输入、输出维度不变，左右两边共用模型的输入，输出时将 PLM 与 `A-B` 的输出叠加；

- 用随机高斯分布 $N(0,\sigma^2)$ 初始化 `A`，用全零矩阵初始化 `B`。矩阵 `B` 的全零初始化，使得在训练最开始的一段时间，右路的结果会接近于0，这样模块的输出就基本上来自于左路，也就是大模型原有参数的计算结果，这使得模型优化的初始点就和原始的大模型保持一致。

运行如下命令进行单机多卡分布式训练：

```shell
export CUDA_VISIBLE_DEVICES = 0, 1, 2, 3
torchrun --nproc_per_node=4  train_lora.py \
--model_name_or_path /data/wuguangshuo/THUDMchatglm-6b/ \
--train_args_file ./config/chatglm_6b_lora.json \
--data_path ./dataset/FreedomIntelligencehuatuo_knowledge_graph_qa/train_huatuo.json \
--max_input_length 256 \
--max_output_length 256
```

训练完成后，运行如下命令进行推理：

```shell
python cli_demo.py \
--model_name_or_path /path/to/chatglm-6b/ \
--lora_checkpoint output/adgen-chatglm-6b-lora/ \
--no_history
```



## Freeze

仅微调模型后几层的代码。

```shell
export CUDA_VISIBLE_DEVICES = 0, 1, 2, 3
torchrun --nproc_per_node=4  train_freeze.py \
--model_name_or_path /data/wuguangshuo/THUDMchatglm-6b/ \
--data_path ./dataset/FreedomIntelligencehuatuo_knowledge_graph_qa/train_huatuo.json \
--max_input_length 256 \
--max_output_length 256
```





1. https://github.com/THUDM/ChatGLM-6B
1. [zejunwang1/chatglm_tuning: 基于 LoRA 和 P-Tuning v2 的 ChatGLM-6B 高效参数微调 (github.com)](https://github.com/zejunwang1/chatglm_tuning)(主要参考对象)
2. https://github.com/liucongg/ChatGLM-Finetuning
3. https://github.com/HarderThenHarder/transformers_tasks/tree/main/LLM
4. https://github.com/hiyouga/ChatGLM-Efficient-Tuning

