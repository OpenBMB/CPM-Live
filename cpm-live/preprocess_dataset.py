import argparse
import multiprocessing
import sys
import time
import random
import numpy as np
from cpm_live.dataset import build_dataset
from cpm_live.tokenizers import CPMAntTokenizer


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = CPMAntTokenizer()

    def convert_to_ids(self, text):
        ids = Encoder.tokenizer.encode(text)
        ids = [j for j in ids if j != Encoder.tokenizer.unk_id]
        return ids

    def encode(self, line):
        data = line.strip()
        data = data.replace("<n>", "\n")
        doc_ids = self.convert_to_ids(data)
        contexts = []
        i = 0
        while i < len(doc_ids):
            if random.random() <= 0.5:
                task = 1
            else:
                task = 2

            piece = doc_ids[i : i + self.args.max_length - self.args.prompt_length - 2]

            if len(piece) < 32:
                break

            i += len(piece)

            if task == 1:
                context = (
                    [1, 1, len(piece) + 2, 1, 0]
                    + [Encoder.tokenizer.bos_id]
                    + piece
                    + [Encoder.tokenizer.eos_id]
                )
            else:
                context = (
                    [2, 1, len(piece) + 2, 2, 1]
                    + [Encoder.tokenizer.bos_id]
                    + piece
                    + [Encoder.tokenizer.eos_id]
                )

            contexts.append(context)
        return contexts, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, help="Path to input TXT", required=True)
    group.add_argument("--max-length", type=int, default=1024, help="The max sequence length")
    group.add_argument("--prompt-length", type=int, default=32, help="The prompt sequence length")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output_path", type=str, required=True)
    group.add_argument("--output_name", type=str, help="Binary output file name", required=True)

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=64, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log_interval", type=int, default=10000, help="Interval between progress updates"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    startup_start = time.time()
    fin = open(args.input, "r", encoding="utf-8")
    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)

    print(f"Output file name: {args.output_name}")

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0

    print("Time to startup:", startup_end - startup_start)
    with build_dataset(args.output_path, args.output_name) as writer:

        for i, (pair_ids, bytes_processed) in enumerate(encoded_docs, start=1):
            if pair_ids is None:
                continue
            total_bytes_processed += bytes_processed
            for pids in pair_ids:
                writer.write(np.array(pids))
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(
                    f"Processed {i} documents",
                    f"({i/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr,
                )

    pool.close()


if __name__ == "__main__":
    main()
