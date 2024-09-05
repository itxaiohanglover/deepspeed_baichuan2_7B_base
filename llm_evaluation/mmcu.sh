
# the base model file path
base_model_path="/hy-tmp/autodl-tmp/artboy/finetune/llm_code/output/baichuan2-sft-1e5-1123-1806/final"
# save path
mmcu_save_dir="./result/mmcu/"
# specify cpu index
cuda_visibale_gpu_index=0

CUDA_VISIBLE_DEVICES=$cuda_visibale_gpu_index python mmcu.py \
    --data_dir ./eval_data/MMCU0513 \
    --ntrain 0 \
    --save_dir $mmcu_save_dir \
    --base_model_path $base_model_path \
