import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# 项目根目录（scripts 的上一级），使用相对路径方便克隆使用
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from swift import SftArguments, sft_main

sft_args = SftArguments(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    dataset=os.path.join(project_root, 'dataset', 'train.jsonl'),#数据位置
    max_length=256,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    output_dir=os.path.join(project_root, 'output', 'cpu_run'),#输出文件位置
    logging_steps=1,
    save_steps=5,
    torch_dtype='float32',
    tuner_type='lora',
    lora_rank=8,
)

print("开始 CPU 微调测试...")
result = sft_main(sft_args)
print(f"🎉 CPU 环境流程跑通！最佳模型检查点：{result['best_model_checkpoint']}")