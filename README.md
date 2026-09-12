# Qwen2.5-0.5B LoRA 微调学习项目

## 项目简介
第一次在魔搭社区 CPU 环境完成 LoRA 微调的全流程记录。

## 环境
- 平台：魔搭社区 Notebook
- 镜像：ubuntu22.04-py312-torch2.3.1-1.40.0
- 框架：ms-swift 4.5.3
- 基础模型：Qwen/Qwen2.5-0.5B-Instruct

## 训练参数
- tuner_type: lora
- lora_rank: 8
- max_length: 256
- per_device_train_batch_size: 1
- gradient_accumulation_steps: 4
- num_train_epochs: 1
- torch_dtype: float32

## 目录结构
- scripts/train_lora.py：训练脚本
- scripts/infer_lora.py：微调前后推理对比脚本
- dataset/train.jsonl：训练数据
- output/checkpoint-580/：LoRA 权重文件
- output/logging.jsonl：训练日志

## 微调前后对比
（可粘贴你之前观察到的回答差异）

## 学习心得
（简单写几句）