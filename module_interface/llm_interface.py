#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from common import constants
from utils import model_utils
from transformers import AutoTokenizer
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer,GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(constants.LLM_MODULE_MODEL_PATH, device_map="auto", trust_remote_code=True, 
                                          torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(constants.LLM_MODULE_MODEL_PATH, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.float16)
model.generation_config = GenerationConfig.from_pretrained(constants.LLM_MODULE_MODEL_PATH)


def get_predict_answer(input_text):
    with torch.autocast("cuda"):
        try:
            # 输入内容处理
            messages = [{"role": "user", "content": input_text}]
            # LLM 生成结果
            response = model.chat(tokenizer, messages)
            response = response.strip(',，')
            res_mem = response.split('\n')
            if len(res_mem) > 1:
               response = '\n'.join(res_mem[1:])
        except Exception as e:
            print(e)
            print('llm has error')
            response = constants.BACK_AND_FORTH_ANSWER
        return response


def get_llm_answer_request(request_obj):
    query = request_obj.get(constants.QUERY_STR)

    predict_text = get_predict_answer(query)
    if predict_text:
        ans_list = [{constants.INDEX_STR: 0,
                     constants.BOT_TYPE: 'LLM',
                     constants.BOT_PRIORITY_STR: 0,
                     constants.ANSWER_STR: predict_text,
                     constants.CONFIDENCE_STR: 100
                     }]
        request_obj[constants.RESPONSE_ANSWER] = ans_list
        return True, request_obj
    else:
        return False, None


def get_llm_answer(query):
    predict_text = get_predict_answer(query)
    if predict_text:
        return True, predict_text
    else:
        return False, None

