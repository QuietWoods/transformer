# -*- coding: utf-8 -*-
# @Time    : 2018/11/1 17:14
# @Author  : QuietWoods
# @FileName: make_corpora.py
# @Software: PyCharm
"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose

from utils import count_data
from metric import compute_rouge_l


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

train_source = open("corpora/train.source-target.source", 'w', encoding='utf-8')
train_target = open("corpora/train.source-target.target", 'w', encoding='utf-8')

test_source = open("corpora/test.source-target.source", 'w', encoding='utf-8')
test_target = open("corpora/test.source-target.target", 'w', encoding='utf-8')

patents = 10

def _split_words(texts):
    return map(lambda t: t.split(), texts)


def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(compute_rouge_l(reference=abst, mode='r'),
                          art_sents))
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores


def process(split, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['article'])
    abs_sents = tokenize(data['abstract'])
    if len(abs_sents) <= 0:
        return
    if art_sents and abs_sents: # some data contains empty article/abstract
        extracted, scores = get_extract_label(art_sents, abs_sents)
    else:
        extracted, scores = [], []
    #print("extracted")
    #print(extracted)
    extracted_sents = [" ".join(art_sents[int(i)]) for i in extracted]
    #print("ext_sents")
    #print(extracted_sents)
    # 关键句集合
    # 关键句对应的摘要句集合
    #print("abs_sents")
    #print(abs_sents)
    abs_sents = [" ".join(sent) for sent in abs_sents]
    # 确保句对完整
    abs_len = len(abs_sents)
    ext_len = len(extracted_sents)
    min_length = min(abs_len, ext_len)
    if split == 'val':
        test_source.write('\n'.join(extracted_sents[:min_length]))
        test_target.write('\n'.join(abs_sents[:min_length]))
    else:     
        train_source.write('\n'.join(extracted_sents[:min_length]))
        train_target.write('\n'.join(abs_sents[:min_length]))


def label_mp(split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def label(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        process(split, i)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main():
    for split in ['val', 'train']:  # no need of extraction label when testing
        label(split)


if __name__ == '__main__':
    main()
    train_source.close()
    train_target.close()
    test_source.close()
    test_target.close()

