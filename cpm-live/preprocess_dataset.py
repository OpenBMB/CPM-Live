import argparse
import multiprocessing
import sys
import time
import random
import json
import numpy as np
from cpm_live.dataset import build_dataset
from cpm_live.tokenizers import CPMAntPlusTokenizer


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = CPMAntPlusTokenizer()

    def convert_to_ids(self, text):
        ids = Encoder.tokenizer.encode(text)
        ids = [j for j in ids if j != Encoder.tokenizer.unk_id]
        return ids

    def encode(self, line):
        data = json.loads(line)
        task = data["task"]
        contexts = []
        if task == "lm":
            doc_ids = self.convert_to_ids(data["text"])
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
        elif task == "mt":
            english = data["english"]
            chinese = data["chinese"]
            en_ids = (
                [Encoder.tokenizer.bos_id]
                + self.convert_to_ids(english)
                + [Encoder.tokenizer.eos_id]
            )
            zh_ids = (
                [Encoder.tokenizer.bos_id]
                + self.convert_to_ids(chinese)
                + [Encoder.tokenizer.eos_id]
            )
            if len(en_ids + zh_ids) > self.args.max_length - self.args.prompt_length:
                return None, 0

            if random.random() <= 0.5:
                # en-zh
                task = 5
            else:
                # zh-en
                task = 6

            if task == 5:
                context = [task, 2, len(en_ids), 1, task, len(zh_ids), 2, task] + en_ids + zh_ids
            else:
                context = [task, 2, len(zh_ids), 1, task, len(en_ids), 2, task] + zh_ids + en_ids

            contexts.append(context)
        elif task == "sum":
            passage = data["passage"]
            abstract = data["abstract"]

            pids = (
                [Encoder.tokenizer.bos_id]
                + self.convert_to_ids(passage)
                + [Encoder.tokenizer.eos_id]
            )
            aids = (
                [Encoder.tokenizer.bos_id]
                + self.convert_to_ids(abstract)
                + [Encoder.tokenizer.eos_id]
            )
            task = 3
            context = [task, 2, len(pids), 1, task, len(aids), 2, task] + pids + aids
            if len(pids + aids) > self.args.max_length - self.args.prompt_length:
                return None, 0
            contexts.append(context)
        elif task == "qa":
            passage = data["passage"]
            q_list = data["question_list"]
            a_list = data["answer_list"]
            assert len(q_list) == len(a_list)
            sep_id = 3
            pids = (
                [Encoder.tokenizer.bos_id]
                + self.convert_to_ids(passage)
                + [Encoder.tokenizer.eos_id]
            )
            qids = [Encoder.tokenizer.bos_id]
            for idx, q in enumerate(q_list):
                qids += self.convert_to_ids(q)
                if idx != len(q_list) - 1:
                    qids.append(sep_id)
            qids += [Encoder.tokenizer.eos_id]

            aids = [Encoder.tokenizer.bos_id]
            for idx, a in enumerate(a_list):
                aids += self.convert_to_ids(a)
                if idx != len(a_list) - 1:
                    aids.append(sep_id)
            aids += [Encoder.tokenizer.eos_id]

            task = 4
            context = (
                [task, 3, len(pids), 1, task, len(qids), 2, task, len(aids), 3, task]
                + pids
                + qids
                + aids
            )
            if len(pids + qids + aids) > self.args.max_length - self.args.prompt_length:
                return None, 0
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
