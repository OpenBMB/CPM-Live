# flake8: noqa
from cpm_live.generation import CPMAntPlusEnZhTranslation, CPMAntPlusZhEnTranslation
from cpm_live.models import CPMAntPlusTorch, CPMAntConfig
from cpm_live.tokenizers import CPMAntPlusTokenizer
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-bminf", default=False, action="store_true",
                       help="Whether to use BMInf")
    parser.add_argument("--memory-limit", type=int, default=12,
                        help="GPU Memory limit, in GB")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    
    texts = ["Treading the fine line between fantasy and reality, David produced his first series of etchings in Berlin while at the same time painting this snapshot of social decay in the growing city that can be sensed nearby."]

    config = CPMAntConfig.from_json_file("config/cpm-ant-plus-10b.json")
    ckpt_path = "YOUR_PATH/cpm-ant-plus-10b.pt"
    model = CPMAntPlusTorch(config=config)

    model.load_state_dict(torch.load(ckpt_path))
    if args.use_bminf:
        import bminf
        model = bminf.wrapper(model, quantization=False, memory_limit=args.memory_limit << 30)
    else:
        model.cuda()
    tokenizer = CPMAntPlusTokenizer()

    # use beam search
    enzh_trans = CPMAntPlusEnZhTranslation(
        model=model,
        tokenizer=tokenizer,
        prompt_length=config.prompt_length
    )
    inference_results = enzh_trans.generate(texts, max_length=300, repetition_penalty=1.2)
    for res in inference_results:
        print(res)
    

    texts = ["即便在 2022 年的今天，推荐系统领域依然有许多悬而未决的问题。"]
    zhen_trans = CPMAntPlusZhEnTranslation(
        model=model,
        tokenizer=tokenizer,
        prompt_length=config.prompt_length
    )
    inference_results = zhen_trans.generate(texts, max_length=300, repetition_penalty=1.2)
    for res in inference_results:
        print(res)


