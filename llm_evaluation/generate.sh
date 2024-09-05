# the base model file path
base_model_path="/hy-tmp/autodl-tmp/artboy/finetune/llm_code/output/baichuan2-sft-1e5-1123-1806/final"
# the test file path
test_file="./eval_data/domain/psychology_test.json"
# the test file output file
output_file="./result/psychology_result.txt"
# specify the gpu
cuda_visibale_gpu_index=0

CUDA_VISIBLE_DEVICES=$cuda_visibale_gpu_index python generate.py \
  --dev_file $test_file \
  --model_name_or_path  $base_model_path\
  --dev_batch_size 8 \
  --output_file $output_file \
