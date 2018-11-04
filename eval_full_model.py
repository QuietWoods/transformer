# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 15:34
# @Author  : QuietWoods
# @FileName: eval_full_model.py
# @Software: PyCharm
""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import os
from os.path import join, exists

from evaluate import eval_meteor, eval_rouge


try:
    _DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def main(args):
    dec_dir = args.decode_file

    ref_dir = args.ref_file
    assert exists(ref_dir)

    if args.rouge:
        output = eval_rouge(dec_dir, ref_dir)
        metric = 'rouge'
    else:
        output = eval_meteor(dec_dir, ref_dir)
        metric = 'meteor'
    print(output)
    with open('{}.txt'.format(metric), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')

    parser.add_argument('--decode_file', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--ref_file', action='store', required=True,
                        help='directory of decoded summaries')

    args = parser.parse_args()
    main(args)

