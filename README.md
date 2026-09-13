<div align="center">

# 🚀 Qwen2.5-0.5B LoRA 微调学习项目

**第一次在魔搭社区 CPU/GPU 环境中完成 LoRA 微调的全流程记录**

![Framework](https://img.shields.io/badge/Framework-ms--swift_4.5.3-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--0.5B--Instruct-green)
![Platform](https://img.shields.io/badge/Platform-ModelScope_Notebook-orange)
![Status](https://img.shields.io/badge/Status-Learning_Project-yellow)

</div>

---

## 📖 项目简介

这是我第一次在**魔搭社区 CPU&GPU 环境**中完成 LoRA 微调的全流程记录。
项目使用 `ms-swift 4.5.3` 框架，对 `Qwen/Qwen2.5-0.5B-Instruct` 进行了轻量级 LoRA 微调，并保存了训练脚本、推理脚本、数据集样例和最终权重，便于复现和回顾。
CPU和GPU环境除了训练硬件不同，其余内容全部保持一致，旨在比较两种不同训练环境的区别

---

## 🛠️ 环境信息

| 项目        | 配置                                      |
| :--------- | :------------------------------------     |
| 🖥️ 平台    | 魔搭社区 Notebook，免费 CPU 环境&限时GPU环境 |
| 📦 镜像    | `ubuntu22.04-py312-torch2.3.1-1.40.0`     |
| ⚙️ 框架    | `ms-swift==4.5.3`                         |
| 🤖 基础模型| `Qwen/Qwen2.5-0.5B-Instruct`               |
| 🐍 Python  | 3.12                                      |
| 🔥 PyTorch | 2.3.1                                     |

---

## ⚙️ 训练参数

| 参数 | 值 |
|:---|:---|
| `tuner_type` | `lora` |
| `lora_rank` | 8 |
| `max_length` | 256 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| `num_train_epochs` | 1 |
| `torch_dtype` | `float32` |
| `logging_steps` | 1 |
| `save_steps` | 5 |

**📊 数据与训练概况**

- 📁 数据集：`dataset/train.jsonl`，原始 **3000** 条，过滤后 **2319** 条（**样本长度超过了你在脚本里设置的 `max_length=256`**）
- 🔄 总步数：**580**
- 💾 最终 checkpoint：`output/checkpoint-580`

---

## 📂 目录结构

```text
lora_project/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   └── train.jsonl
├── scripts/
│   ├── train_lora.py
│   └── infer_lora.py
└── output/
    ├── checkpoint-580/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── logging.jsonl
```

> 💡 **说明**：训练状态文件如 `optimizer.pt`、`scheduler.pt` 等未上传，只保留了 LoRA 核心权重和日志。

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```


### 2️⃣ 训练（CPU 环境验证）

```bash
python scripts/train_lora.py
```

### 3️⃣ 推理对比（微调前 vs 微调后）

```bash
python scripts/infer_lora.py
```

---

## 🔄 微调前后模型推理结果对比

> 使用相同问题：**"你好，请介绍一下你自己"**

### 🔹 微调前（基础模型）

> 您好！我是Qwen，由阿里云开发的超大规模语言模型。我叫通义千问，是阿里巴巴集团研发的语言模型，可以回答问题、创作文字和图片内容，并且能够进行对话交流。我的目标是帮助用户更好地理解和解决问题。如果您有任何问题或需要帮助，请随时告诉我，我会尽力提供支持。

### 🔸 微调后（LoRA 模型）

> 您好！我是一个人工智能助手，由阿里云开发。我是通过学习大量的文本数据来训练的，并且能够回答问题、提供信息和进行对话。我的目标是帮助用户更好地理解和解决问题。如果您有任何问题或需要帮助，请随时告诉我。

### 🔍 微调前后的具体差异

|维度|微调前（基础模型）|微调后（LoRA）|变化分析|
|---|---|---|---|
|**自我认知**|我叫“通义千问”，是“阿里巴巴集团”研发的语言模型。|我是一个“人工智能助手”，由“阿里云”开发。|**核心变化**：模型放弃了原有的品牌特化身份，接受了训练数据中设定的通用助手身份。|
|**能力描述**|可以回答问题、**创作文字和图片内容**。|通过学习大量的文本数据来训练，能够回答问题、提供信息和进行对话。|**减少幻觉**：基础模型声称能“生成图片”（0.5B纯文本模型其实做不到），微调后表述更严谨，贴合纯文本助手的定位。|
|**语气风格**|偏向官方、品牌宣传。|更偏向通用、平实的助手口吻。|风格向你的训练数据靠拢。|
|**结尾**|几乎完全一致（“我的目标是帮助用户更好地理解和解决问题……”）|几乎完全一致|说明基础模型的安全回复模板还保留着，微调没有破坏原有基础能力。|

### ✅ 为什么说微调是成功的？

1. **模型“听话”了**：我希望它变成“通用AI助手”的身份，它确实做到了。这说明 LoRA 权重成功地将训练数据的风格注入到了基础模型中。
    
2. **没有崩溃**：模型回答逻辑通顺，没有出现乱码、重复或胡言乱语，说明学习率、LoRA秩等参数设置合理。
    
3. **保留了通用能力**：虽然身份变了，但回答问题的逻辑和礼貌性没有丢失。

### ⚠️ 为什么变化看起来“不够大”？

原因有几个：

1. **模型太小**：Qwen2.5-0.5B 本身容量有限，能记住“我是谁”这种浅层身份，但很难彻底改变深层的语言习惯。
    
2. **数据量偏少**：2319 条数据，对于改变一个模型的身份认知来说，已经够用，但不足以让它完全变成“另一个人”。
    
3. **LoRA秩较低**： `lora_rank=8`，这是一个很保守的秩。它只够学习“表面风格”，如果想让模型学得更深，可以尝试 `lora_rank=16` 或 `32`。
    
4. **训练轮数少**：`num_train_epochs=1` 只过了一遍数据，模型只是“浅尝辄止”。


---

## 💡 学习心得

- ✅ 第一次完整走通了 **LoRA 微调（CPU&GPU）全流程**：数据准备、环境配置、脚本编写、训练、保存、推理验证。

- ⚠️ **踩坑记录**：`ms-swift 4.x` 版本 API 有较大变化：
  - `SftArguments` 和 `sft_main` 需从 `swift` 顶层导入
  - `use_lora` 已被 `tuner_type` 替代

- 📦 理解了 LoRA 权重的保存结构，核心文件只有 `adapter_config.json` 和 `adapter_model.safetensors`。

- 🐢 GPU VS CPU 
### 1. 核心指标对比
| 对比维度 | CPU 训练 | GPU 训练 |
| :--- | :--- | :--- |
| **总耗时** | 2931.4518s（≈48分 51秒） | 608.142s（≈10分 8秒） |
| **训练速度** | 1x（基准） | ~5x（约为 CPU 的 5 倍） |
| **数据类型** | `float32` | `bfloat16` |
| **Loss 曲线** | 保持一致 | 保持一致 |
### 2. 性能与耗时分析
* **加速比符合预期**：GPU 训练速度大致是 CPU 的 5 倍。由于本次实验为**小模型**且 **`batch_size=1`**，GPU 的大规模并行计算优势无法尽情展现，因此 5 倍的加速比在合理预期范围内。
* **Loss 曲线一致性**：无论在 CPU 还是 GPU 环境下，两者的 loss 曲线均保持一致，证明训练过程收敛稳定，未因硬件或数据类型的改变而产生异常。
### 3. 数据类型 (dtype) 差异说明
本次实验中，CPU 与 GPU 采用了不同的数据类型，这是基于硬件特性的合理选择：
* **CPU (`float32`)**：
  * CPU 对 `bfloat16` 的支持较差，运算效率低。
  * `float32` 具有极佳的通用性，且计算稳定，是 CPU 训练的常规首选。
* **GPU (`bfloat16`)**：
  * 现代 NVIDIA GPU 具备 Tensor Core，原生支持 `bfloat16` 计算，能够有效减半显存占用并大幅提升吞吐量。
  * `bfloat16` 与 `float32` 具有相同的动态范围，在降低精度的同时，能有效避免训练过程中的数值溢出问题，兼顾了效率与稳定性。


---

## 🎯 后续计划

- [ ] 在 GPU 环境中使用更大的模型（如 `Qwen2.5-3B` 或 `7B`）进行微调。
- [ ] 尝试更大的 `lora_rank` 和更多训练轮数、batch_size等参数，观察效果提升
- [ ] 探索将 LoRA 权重合并到基础模型并部署为在线 Demo

---

<div align="center">

📌 **本项目仅用于学习记录，欢迎交流！**

</div>
