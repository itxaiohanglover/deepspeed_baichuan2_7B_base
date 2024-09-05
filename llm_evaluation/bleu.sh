input_file="./result/psychology_result.txt"
save_file="./result/bleu/psychology_result_bleu.txt"
python bleu.py --input_file $input_file \
  --save_file $save_file
