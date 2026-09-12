import gc
from swift import TransformersEngine, InferRequest, RequestConfig

model_id_or_path = 'Qwen/Qwen2.5-0.5B-Instruct'
lora_checkpoint = '/mnt/workspace/lora_test/output/cpu_run/v2-20260912-152936/checkpoint-580'

question = '你好，请介绍一下你自己'
infer_request = InferRequest(messages=[{'role': 'user', 'content': question}])
request_config = RequestConfig(max_tokens=512, temperature=0)

print(f'问题：{question}\n')

# ===== 1. 微调前：加载基础模型，推理，释放 =====
print('正在加载基础模型（微调前）...')
engine_base = TransformersEngine(model_id_or_path)
print('===== 微调前（基础模型） =====')
resp_base = engine_base.infer([infer_request], request_config)
print(resp_base[0].choices[0].message.content)

# 释放基础模型占用的内存
del engine_base
gc.collect()
print('\n[基础模型已释放]\n')

# ===== 2. 微调后：加载基础模型 + LoRA，推理，释放 =====
print('正在加载微调后模型（基础模型 + LoRA）...')
engine_lora = TransformersEngine(model_id_or_path, adapters=[lora_checkpoint])
print('===== 微调后（LoRA） =====')
resp_lora = engine_lora.infer([infer_request], request_config)
print(resp_lora[0].choices[0].message.content)

# 释放 LoRA 模型占用的内存
del engine_lora
gc.collect()
print('\n[LoRA 模型已释放]')