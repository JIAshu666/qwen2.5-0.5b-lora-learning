import os
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from swift import SftArguments, sft_main

# 自动检测设备
use_gpu = torch.cuda.is_available()

# 根据设备选择精度和输出目录
if use_gpu:
    dtype = 'bfloat16'
    output_dir = os.path.join(project_root, 'output', 'gpu_run')
    # 如果多卡环境，只想用第一张卡，取消下面注释
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    dtype = 'float32'
    output_dir = os.path.join(project_root, 'output', 'cpu_run')

sft_args = SftArguments(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    dataset=os.path.join(project_root, 'dataset', 'alpaca_zh_3k.json'),
    max_length=256,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    output_dir=output_dir,
    logging_steps=1,
    save_steps=5,
	save_total_limit=3,
    dtype=dtype,
    tuner_type='lora',
    lora_rank=8,
)

device_name = "GPU" if use_gpu else "CPU"
print(f"开始 {device_name} 微调...")
result = sft_main(sft_args)
print(f"🎉 {device_name} 环境流程跑通！最佳模型检查点：{result['best_model_checkpoint']}")