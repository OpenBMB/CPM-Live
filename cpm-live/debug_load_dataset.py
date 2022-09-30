from cpm_live.training_tasks.bee import MixedDataset
from cpm_live.tokenizers import CPMBeeTokenizer
import torch
from tqdm import tqdm


def main():
    tokenizer = CPMBeeTokenizer()
    dataloader = MixedDataset("debug.json", 8, 2048, tokenizer, max_depth=8)
    dataloader.start()

    try:
        print("Start loop")
        for data in tqdm(dataloader):
            from IPython import embed

            embed()
            continue
            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_ids_sub = torch.from_numpy(data["inputs_sub"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            input_context = torch.from_numpy(data["context"]).cuda().bool()
            input_sample_ids = torch.from_numpy(data["sample_ids"]).cuda().to(torch.int32)
            input_num_segments = torch.from_numpy(data["num_segments"]).cuda().to(torch.int32)
            input_segment_ids = torch.from_numpy(data["segment_ids"]).cuda().to(torch.int32)
            input_segment_rel_offset = (
                torch.from_numpy(data["segment_rel_offset"]).cuda().to(torch.int32)
            )
            input_segment_rel = torch.from_numpy(data["segment_rel"]).cuda().to(torch.int32)
            input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            ext_table_ids = torch.from_numpy(data["ext_ids"]).cuda().to(torch.int32)
            ext_table_sub = torch.from_numpy(data["ext_sub"]).cuda().to(torch.int32)
            task_ids = torch.from_numpy(data["task_ids"]).cuda().to(torch.int32)
            task_names = data["task_names"]

            # embed()
            # break
    finally:
        dataloader.close()


if __name__ == "__main__":
    main()
