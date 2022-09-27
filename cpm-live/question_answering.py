# flake8: noqa
from cpm_live.generation import CPMAntPlusQuestionAnswering
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
        [
        """此外，宋雯还介绍了党的十八大以来，我国能源基础设施建设取得的成就。她表示，经过多年发展，我国已成为世界能源生产第一大国，构建了多元清洁的能源供应体系，形成了横跨东西、纵贯南北、覆盖全国、连通海外的能源基础设施网络，有力保障了经济社会发展用能需求，主要体现在4个方面：一是保障能源供应的基础设施更加完善。建成全球规模最大的电力系统，发电装机达到24.7亿千瓦，超过G7国家装机规模总和；35千伏及以上输电线路长度达到226万公里，建成投运特高压输电通道33条，西电东送规模接近3亿千瓦，发电装机、输电线路、西电东送规模分别比十年前增长了1.2倍、0.5倍、1.6倍。油气“全国一张网”初步形成，管网规模超过18万公里，比十年前翻了一番，西北、东北、西南和海上四大油气进口战略通道进一步巩固。十年来，能源生产以年均约2.4%的增长支撑了国民经济年均6.6%的增长，能源自给率长期稳定在80%以上。"""
    , "我国能源自给率是多少？", "35千伏及以上输电线路长度达到多少公里？", "发电装机是多少", "西电东送规模是多少", "我国是世界能源生产第几大国"]
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
    question_answering = CPMAntPlusQuestionAnswering(
        model=model,
        tokenizer=tokenizer,
        prompt_length=config.prompt_length
    )
    inference_results = question_answering.generate(texts, max_length=300)
    for res in inference_results:
        print(res)
