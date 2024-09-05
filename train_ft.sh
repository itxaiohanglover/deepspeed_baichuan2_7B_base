CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --num_gpus=3 --master_port 65224 train.py --train_args_file /hy-tmp/autodl-tmp/artboy/finetune/llm_code/training_config/baichuan2_config.json
