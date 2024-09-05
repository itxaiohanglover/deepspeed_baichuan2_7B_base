import sys
import argparse
import json
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer,GenerationConfig


sys.path.append("../")
from utils import model_utils


def load_dev_data(dev_file_path):
    dev_data = []
    with open(dev_file_path) as f:
        lines = f.readlines()
        for line in lines:
            dev_data.append(json.loads(line.strip()))
    print(dev_data[:10])
    return dev_data


def generate_text(dev_data, batch_size, tokenizer, model, output_path):
    ofile = open(output_path, 'a', encoding="utf-8")
    for i in tqdm(range(0, len(dev_data), batch_size), total=len(dev_data)//batch_size, unit="batch"):
        batch = dev_data[i:i+batch_size]
        for item in batch:
            # 拼接输入信息
            input_text = item['instruction'] + item['input']
            messages = [{"role": "user", "content": input_text}]
            # LLM 生成结果
            response = model.chat(tokenizer, messages)
            # 结果格式化输出
            ofile.write(json.dumps({"input": input_text, "predict": response, "target": item["output"]},
                                   ensure_ascii=False) + "\n")


def main(args):
    with torch.autocast("cuda"):
        dev_data = load_dev_data(args.dev_file)[args.start_index:]
        generate_text(dev_data, batch_size, tokenizer, model, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="pretrained language model")
    parser.add_argument("--max_length", type=int, default=512, help="max length of dataset")
    parser.add_argument("--dev_batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--output_file", type=str, default="data_dir/predictions.json")
    parser.add_argument("--start_index", type=int, default=0, help="test file start index")

    args = parser.parse_args()
    batch_size = args.dev_batch_size
    model_dir = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,
                                              torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,
                                                 torch_dtype=torch.float16)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    main(args)
