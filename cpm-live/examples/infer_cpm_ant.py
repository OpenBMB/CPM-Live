import sys

sys.path.insert(0, "..")
import os
import json
import torch
from opendelta import LoraModel
from task_config import task_config
from arguments import get_args
from cpm_live.models import CPMAntTorch, CPMAntConfig
from cpm_live.tokenizers import CPMAntTokenizer


if __name__ == "__main__":
    args = get_args()

    # init model
    config = CPMAntConfig.from_json_file(args.config_path)
    model = CPMAntTorch(config=config)
    # insert LoRA
    delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"])
    delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
    # load checkpoint
    model.load_state_dict(torch.load(args.model_path), strict=False)
    # load delta weights
    model.load_state_dict(torch.load(os.path.join(args.output_path, "best.pt")), strict=False)
    model.cuda()
    delta_model.log()

    print("[INFO] inference begins...")
    config_dict = task_config[args.dataset_name]
    tokenizer = CPMAntTokenizer()
    infer = config_dict["infer"](
        model=model,
        tokenizer=tokenizer,
        prompt_length=config.prompt_length,
    )

    results = []
    with open(os.path.join(args.dataset_path, "test.jsonl"), "r") as f:
        for line in f:
            results.extend(
                infer.generate(
                    [json.loads(line)], max_length=args.infer_maxlen, cls_num=args.cls_num
                )
            )

    with open(os.path.join(args.output_path, "infer_result.json"), "w") as out:
        json.dump(results, out, ensure_ascii=False)

    print("[INFO] inference is finished!")
