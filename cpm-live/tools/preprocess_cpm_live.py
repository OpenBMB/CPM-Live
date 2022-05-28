# coding=utf-8

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

from cpm_live_tokenizer import CPMLiveTokenizer
from model_center.tools import indexed_dataset

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = CPMLiveTokenizer(os.path.join(self.args.tokenizer_path, 'vocab.txt'))

    def convert_to_ids(self, text):
        ids = Encoder.tokenizer.encode(text)
        ids = [j for j in ids if j != Encoder.tokenizer.unk_id]
        return ids

    def encode(self, line):
        data = line.strip()
        data = data.replace("<n>", "\n")
        doc_ids = Encoder.tokenizer.encode(data)
        doc_ids = [j for j in doc_ids if j != Encoder.tokenizer.unk_id]
        contexts = []
        i = 0
        while i < len(doc_ids):
            piece = doc_ids[i:i+478]
            if len(piece) < 32:
                break
            i += 478
            
            context = [Encoder.tokenizer.bos_id] + piece + [Encoder.tokenizer.eos_id]
            assert len(context) - len(piece) == 2
            contexts.append(context)
        return contexts, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', default="../train_0.txt", type=str, help='Path to input TXT')
    
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="vocab", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="../data_bin/", type=str)
    group.add_argument('--output_prefix', default="cpm_live_text_0", type=str, help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=32,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

def main():
    args = get_args()
    startup_start = time.time()
    fin = open(args.input, 'r', encoding='utf-8')
    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)

    print(f"Output prefix: {args.output_prefix}")
    
    context_bin_file = os.path.join(args.output_path, "{}_context_{}.bin".format(args.output_prefix, '0'))
    context_idx_file = os.path.join(args.output_path,  "{}_context_{}.idx".format(args.output_prefix, '0'))
    builder_context = indexed_dataset.make_builder(context_bin_file, impl=args.dataset_impl, dtype=np.uint16)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0

    print("Time to startup:", startup_end - startup_start)

    for i, (pair_ids, bytes_processed) in enumerate(encoded_docs, start=1):
        if pair_ids is None:
            continue
        total_bytes_processed += bytes_processed
        for pids in pair_ids:
            builder_context.add_item(torch.IntTensor(pids))
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    builder_context.finalize(context_idx_file)
    pool.close()

if __name__ == '__main__':
    main()
