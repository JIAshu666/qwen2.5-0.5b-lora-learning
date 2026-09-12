import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 显式指定不使用GPU

from swift import SftArguments, sft_main  # ✅ 正确导入：sft_main 和 SftArguments

# 定义SFT微调训练参数
sft_args = SftArguments(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    dataset='/mnt/workspace/lora_test/dataset/alpaca_zh_3k.json',
    max_length=256,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    output_dir='/mnt/workspace/lora_test/output/cpu_run',
    logging_steps=1,
    save_steps=5,
    dtype='float32',
    tuner_type='lora',   # 指定 LoRA 微调
    lora_rank=8,
)

print("开始 CPU 微调测试...")
result = sft_main(sft_args)  # ✅ 使用 sft_main 启动训练
print(f"🎉 CPU 环境流程跑通！最佳模型检查点：{result['best_model_checkpoint']}")