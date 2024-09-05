import torch
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer,GenerationConfig

#model_dir = "/hy-tmp/autodl-tmp/artboy/base_model/Baichuan2-7B-Base"
model_dir = "/hy-tmp/autodl-tmp/artboy/finetune/llm_code/output/baichuan2-sft-1e5-1123-1806/final"
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)
model.generation_config = GenerationConfig.from_pretrained(model_dir)


print("Human:")
line = input()
while line:
        messages = []
        messages.append({"role": "user", "content": line})
        response = model.chat(tokenizer, messages)
        response = response.strip(',，')
        res_mem = response.split('\n')
        if len(res_mem) > 1:
            response = '\n'.join(res_mem[1:])
        print("Assistant:\n" + response)
        print("\n------------------------------------------------\nHuman:")
        line = input()
