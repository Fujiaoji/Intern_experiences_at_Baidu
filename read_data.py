import argparse
import os
from functools import partial

import paddle
from paddlenlp.data import Stack, Tuple, Pad

from data import load_dataset, load_dict
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification, ErnieModel

from paddlenlp.datasets import MapDataset

# 读取数据train dev test
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            words = []
            arcs = []
            rels = []
            for sid, line in enumerate(fp.readlines()):
                if sid % 3 == 0:
                    words = line.strip().split('\t')
                elif sid % 3 == 1:
                    arcs = line.strip().split('\t')
                    arcs = [int(i) for i in arcs]
                elif sid % 3 == 2:
                    rels = line.strip().split('\t')
                    
                    yield words, arcs, rels
    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

# 读取rels的vocab
def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            key = line.strip('\n')
            vocab[key] = i
            i += 1
    return vocab

def convert_to_features(example, tokenizer, rels_vocab):
    tokens, arcs, rels = example
    tokenized_input = tokenizer(
        tokens, return_length=True, is_split_into_words=True)
    # Token '[CLS]' and '[SEP]' will get label 'HED'
    rels = ['[CLS]'] + rels + ['[SEP]']#给vocab里要加上CLS与SEP
    tokenized_input['rels'] = [rels_vocab[x] for x in rels]
    tokenized_input['arcs'] = [0] + arcs + [1] # 加了个开头结尾，虽然1会有冲突，但是maks掉了，所以没关系；
    return tokenized_input['input_ids'], tokenized_input['token_type_ids'], tokenized_input['seq_len'], tokenized_input['rels'], tokenized_input['arcs']

if __name__ == "__main__":
    data_dir = "../data/"

    train_ds, dev_ds, test_ds = load_dataset(
                datafiles=(os.path.join(data_dir, 'train.txt'),
                        os.path.join(data_dir, 'dev.txt'), 
                        os.path.join(data_dir, 'test.txt')))
    rels_vocab = load_dict(os.path.join(data_dir, 'rels_vocab.txt'))

    tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
    
    trans_func = partial(
        convert_to_features, tokenizer=tokenizer, rels_vocab=rels_vocab)

    train_ds.map(trans_func)
    dev_ds.map(trans_func)
    test_ds.map(trans_func)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # token_type_ids
        Stack(dtype='int64'),  # seq_len
        Pad(axis=0, pad_val=rels_vocab.get("HED", 2), dtype='int64'),  # rels
        Pad(axis=0, pad_val=0, dtype='int32') # arcs
    ): fn(samples)

    # 自动padding到batch里最大的长度
    train_loader = paddle.io.DataLoader(
            dataset=train_ds,
            batch_size=4, #args.batch_size,
            return_list=True,
            collate_fn=batchify_fn)
    
    for example in train_loader:
        print(example[0])

