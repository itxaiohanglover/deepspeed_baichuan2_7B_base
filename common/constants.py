
# =======================Request=============================
QUERY_STR = 'query'

# =======================Http=============================
HTTP_SERVER = 'http://127.0.0.1:10820/internal'
ASK_STR = 'ask'
API_TYPE_STR = 'apiType'
ERR_MSG_STR = 'errMsg'
FEEDBACK_STR = 'user_feedback'
API_TYPE_ERROR_STR = 'apiType is error!'

# =======================Response============================
RESPONSE_ANSWER = 'answer'
BOT_TYPE = 'bot_type'
BOT_PRIORITY_STR = 'bot_priority'
INDEX_STR = 'index'
ANSWER_STR = 'answer'
CONFIDENCE_STR = 'confidence'

# =======================LLM Module==========================
#LLM_MODULE_MODEL_PATH = '/root/autodl-tmp/artboy/base_model/Baichuan2-7B-Chat'
LLM_MODULE_MODEL_PATH = '/hy-tmp/autodl-tmp/artboy/finetune/llm_code/output/baichuan2-sft-1e5-1123-1806/final'
LLM_MODULE_LORA_WEIGHTS = '../trained_models/Baihuan2-SFT'
LLM_MODULE_CUTOFF_LEN = 1024

# ===========================NLG=============================
BACK_AND_FORTH_ANSWER = '不好意思，没有找到你需要的答案，我会继续学习，为你提供满意的答案'
