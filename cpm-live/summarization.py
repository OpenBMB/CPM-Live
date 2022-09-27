# flake8: noqa
from cpm_live.generation import CPMAntPlusSummarization
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

    texts = [
        """一是设施网络化水平不断提高。
规模优势凸显，到2021年底，我国综合交通网总里程突破600万公里，220千伏及以上输电线路79.4万公里，光缆线路总长度达到5481万公里，分别相当于10年前的1.3倍、1.7倍和3.7倍，水库总库容达到9035亿立方米，形成了超大规模网络，高铁、高速公路、电网、4G网络规模等长期稳居世界第一。
结构日趋合理，高铁、高速公路、特高压输电线路、5G网络快速发展，高标准高品质基础设施比例不断提高。以沙漠、戈壁、荒漠地区为重点的清洁能源基地加快建设，新能源装机和发电量比重不断提升，有力促进碳达峰碳中和目标实现。基础性网络不断拓展提升，农村公路10年间净增90多万公里，农村供电网络不断优化提升，2015年消除了无电人口。
二是服务质量能力持续提升。
网络覆盖广，高速铁路对百万人口以上城市覆盖率超过95%，高速公路对20万以上人口城市覆盖率超过98%，民用运输机场覆盖92%左右的地级市，具备条件的建制村实现通硬化路、通宽带、直接通邮，农村自来水普及率提高到84%左右，4G、5G用户普及率达到87%左右。"""
    ]

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
    summarization = CPMAntPlusSummarization(
        model=model,
        tokenizer=tokenizer,
        prompt_length=config.prompt_length
    )
    inference_results = summarization.generate(texts, max_length=300, repetition_penalty=1.2)
    for res in inference_results:
        print(res)
