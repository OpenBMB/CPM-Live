from training_tasks.bee import MixedDataset
from cpm_live.tokenizers import CPMAntTokenizer
import os
import torch
import bmtrain as bmt
import time
from tqdm import tqdm

def main():
    tokenizer = CPMAntTokenizer()
    dataloader = MixedDataset(
        "datasets.json",
        16,
        512,
        tokenizer,
        [0.5, 0.25, 0.25]
    )
    if os.path.exists("data.pt"):
        # load dataset states if exists
        dataset_states = torch.load("data.pt")
        missing = dataloader.load_state_dict(dataset_states)
        if len(missing) > 0:
            bmt.print_rank("Missing keys when loading dataset states: ", missing)
    print("Start dataloader")
    dataloader.start()

    try:
        print("Start loop")
        for data in tqdm(dataloader):
            # print(data)
            input_ids = torch.from_numpy(data["inputs"]).cuda().long()
            length = torch.from_numpy(data["length"]).cuda().long()
            context = torch.from_numpy(data["context"]).cuda().bool()
            sample_ids = torch.from_numpy(data["sample_ids"]).cuda().long()
            num_segments = torch.from_numpy(data["num_segments"]).cuda().long()
            segment_ids = torch.from_numpy(data["segment_ids"]).cuda().long()
            segment_rel_offset = torch.from_numpy(data["segment_rel_offset"]).cuda().long()
            segment_rel = torch.from_numpy(data["segment_rel"]).cuda().long()
            span = torch.from_numpy(data["spans"]).cuda().long()
            targets = torch.from_numpy(data["target"]).cuda().long()
            task_ids = torch.from_numpy(data["task_ids"]).cuda().long()
            task_names = data["task_names"]
            
            batch = input_ids.size(0)
            seqlen = input_ids.size(1)
            with torch.no_grad():
                device = input_ids.device

                # calc segment bucket
                segment_rel_2d = torch.masked_fill(
                    segment_ids[:, :, None] * num_segments[:, :, None] + segment_ids[:, None, :] + segment_rel_offset[:, :, None],
                    ~((sample_ids[:, :, None] == sample_ids[:, None, :]) & (span[:, None, :] == span[:, :, None])), # not in the same span and sample
                    0,  # avoid torch.gather overflow
                ).view(batch, seqlen * seqlen)

                segment_bucket = torch.gather(
                    input = segment_rel,
                    dim = 1,
                    index = segment_rel_2d,
                ).view(batch, seqlen, seqlen)
                
                segment_bucket.masked_fill_(
                    ~((sample_ids[:, :, None] == sample_ids[:, None, :]) & (span[:, None, :] == span[:, :, None])), # not in the same span and sample
                    1,  # bucket is used for in-context samples
                )

                # directional mask
                directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(
                    seqlen, device=device
                ).view(-1, 1)
                # sample mask
                sample_mask_2d = (sample_ids[:, :, None] == 0) | (sample_ids[:, :, None] == sample_ids[:, None, :])
                # context mask
                attention_mask = context[:, None, :] | (
                    context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
                )
                # span mask
                attention_mask = attention_mask & sample_mask_2d & (span[:, None, :] == span[:, :, None])
                # length mask
                mask_1d = (
                    torch.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]
                )
                attention_mask = (
                    mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
                )
            # print("mask: ", attention_mask)
            # print("bucket: ", segment_bucket)
            from IPython import embed; embed()
            break
    finally:
        dataloader.close()

if __name__ == "__main__":
    main()