import sys

sys.path.insert(0, "..")
import os
import bmtrain as bmt
from opendelta import LoraModel
from task_config import task_config
from arguments import get_args
from cpm_live.models import CPMAntPlus, CPMAntConfig
from cpm_live.tokenizers import CPMAntPlusTokenizer


if __name__ == "__main__":
    args = get_args()

    data = {}
    data["train"] = os.path.join(args.dataset_path, "train.jsonl")
    data["eval"] = os.path.join(args.dataset_path, "eval.jsonl")

    # load model
    bmt.init_distributed(seed=0)
    config = CPMAntConfig.from_json_file(args.config_path)
    model = CPMAntPlus(config=config)
    bmt.load(model, args.model_path)
    # insert LoRA
    delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"])
    delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
    delta_model.log()

    bmt.print_rank("[INFO] Tuning begins...")
    tokenizer = CPMAntPlusTokenizer()
    config_dict = task_config[args.dataset_name]
    tune = config_dict["tune"](
        model=model,
        tokenizer=tokenizer,
        prompt_length=config.prompt_length,
        lr=args.lr,
        warmup_iters=args.warmup_iters,
        max_len=args.tune_maxlen,
        cls_num=args.cls_num,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        eval_interval=args.eval_interval,
        output_path=args.output_path,
    )
    tune.run(data)
    bmt.print_rank("[INFO] Tuning is finished!")
