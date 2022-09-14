# flake8: noqa
from cpm_live.generation import CPMAntBeamSearch, CPMAntRandomSampling
from cpm_live.models import CPMAnt, CPMAntConfig
from cpm_live.tokenizers import CPMAntTokenizer
import bmtrain as bmt

if __name__ == "__main__":
    bmt.init_distributed(seed=0)
    texts = [
        "神舟十四号3名航天员在轨期间，各项工作顺利推进。刚刚在轨度过中秋佳节的他们，正在为第二次出舱活动积极准备中，",
        "9月13日，在中共中央宣传部举行的“中国这十年”系列主题新闻发布会上，水利部部长李国英在回答澎湃新闻（www.thepaper.cn）记者提问时表示，通过采取“节、控、换、补、管”等措施，这几年华北地下水超采治理取得了明显成效。",
        "宇宙飞船带着12名狱族源生命，经过层层时空查验，最终降落到这座陆地世界。",
    ]

    config = CPMAntConfig.from_json_file("config/cpm-ant-10b.json")
    model = CPMAnt(config=config)
    ckpt_path = "YOUR_PATH/cpm_live_48_4096_checkpoint-228000.pt"

    bmt.load(model, ckpt_path)
    tokenizer = CPMAntTokenizer()

    # use beam search
    beam_search = CPMAntBeamSearch(
        model=model, tokenizer=tokenizer, prompt_length=config.prompt_length
    )
    inference_results = beam_search.generate(texts)
    for res in inference_results:
        print(res)

    # use random sampling
    beam_search = CPMAntRandomSampling(
        model=model, tokenizer=tokenizer, prompt_length=config.prompt_length
    )
    inference_results = beam_search.generate(texts)
    for res in inference_results:
        print(res)
