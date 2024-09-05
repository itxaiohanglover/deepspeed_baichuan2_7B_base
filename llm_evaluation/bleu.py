import sys
import json
import argparse
from collections import Counter
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu

sys.path.append("../")
from utils import data_utils


weights = [0.25, 0.25, 0.25, 0.25]


def calc_bleu_of_file(json_file, save_file):
    """
    calc bleu of file
    :param json_file:
    :return:
    """
    lines = data_utils.read_big_file_lines(json_file)
    total_bleu = 0
    total_num = 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
        ajson = json.loads(line)
        predict = ajson.get('predict', '')
        target = ajson.get('target', '')
        if predict and target:
            predict_list = list(predict)
            target_list = [list(target)]
            bleu_value = sentence_bleu(target_list, predict_list, weights=weights)
            total_bleu += bleu_value
            total_num += 1
            save_line = ajson.get('input', '').replace('\n', '\\n') + '\t' + predict.replace('\n', '\\n') + '\t' + \
                            target.replace('\n', '\\n') + '\t' + str(bleu_value)
            data_utils.save_line_to_file(save_file, save_line + '\n')
    print('total_bleu:', total_bleu, ', total_num: ', total_num, 'avg bleu: ', total_bleu / total_num)
    data_utils.save_line_to_file(save_file, str(total_bleu / total_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="calc input file")
    parser.add_argument("--save_file", type=str,
                        help="bleu save file")

    args = parser.parse_args()
    save_file = args.save_file
    if args.input_file:
        print('input  file is: ', args.input_file)
        calc_bleu_of_file(args.input_file, save_file)


