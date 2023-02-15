# flake8: noqa
from cpm_live.generation import CPMAntBeamSearch, CPMAntRandomSampling
from cpm_live.models import CPMAntTorch, CPMAntConfig
from cpm_live.tokenizers import CPMAntTokenizer
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

    texts = [
        "我们在假期去了法国的埃菲尔铁塔，",
    ]

    config = CPMAntConfig.from_json_file("YOUR_PATH/cpm-ant-10b.json")
    ckpt_path = "YOUR_PATH/cpm-ant-10b.pt"
    model = CPMAntTorch(config=config)

    model.load_state_dict(torch.load(ckpt_path))
    if torch.cuda.is_available():
        if args.use_bminf:
            import bminf
            model = bminf.wrapper(model, quantization=False, memory_limit=args.memory_limit << 30)
        else:
            model.cuda()
    else:
        model = model.to(torch.float)
    tokenizer = CPMAntTokenizer()

    # use beam search
    beam_search = CPMAntBeamSearch(
        model=model,
        tokenizer=tokenizer,
        prompt_length=config.prompt_length
    )
    inference_results = beam_search.generate(texts, max_length=100, repetition_penalty=1.2)
    for res in inference_results:
        print(res)

    # use top-k/top-p sampling
    random_sample = CPMAntRandomSampling(
        model=model,
        tokenizer=tokenizer,
        prompt_length=config.prompt_length,
    )
    inference_results = random_sample.generate(texts, max_length=100)
    for res in inference_results:
        print(res)
