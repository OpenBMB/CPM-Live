import torch
import torch.utils.data as data
import random
import numpy as np


class CPMLive_Dataset(data.Dataset):
    def __init__(self, ctx, max_length = 1024, prompt_length = 32, tokenizer = None):
        self.ctx = ctx
        self.max_length = max_length + prompt_length
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ctx)

    def __get_item_data(self, raw_data, index):

        global_task = raw_data[0]
        n_segment = raw_data[1]
        len_info = n_segment * 3 + 2
        segment_len  = raw_data[2:len_info:3] 
        segment_type = raw_data[3:len_info:3]
        segment_task = raw_data[4:len_info:3]
        ctx = raw_data[len_info:]

        if ctx.shape[0] > self.max_length - self.prompt_length:
            return None, None, None, None, None, None, None
        len_ctx = min(ctx.shape[0], self.max_length - self.prompt_length)

        context_inp = np.full(len_ctx, True)
        position_inp = np.arange(len_ctx, dtype=np.int64)
        segment_inp = np.full(len_ctx, 0, dtype=np.int64)
        task_inp = np.full(len_ctx, 0, dtype=np.int64)
        tgt = np.full(len_ctx, -100, dtype=np.int64)

        # for each segment
        segment_begin = 0
        for i in range(n_segment):
            segment_end = segment_begin + segment_len[i]
            task = segment_task[i]
            # generate target
            if task == 0:
                num_mask = random.randint(1, segment_len[i] - 1)
                mask_idx = np.random.choice(segment_len[i] - 1, num_mask, replace=False) + segment_begin
                context_inp[mask_idx + 1] = False
                task_inp[segment_begin:segment_end] = task
                assert segment_type[i] == 1
            elif task == 1:
                num_mask = random.randint(1, segment_len[i] - 1)
                context_inp[segment_end-num_mask:segment_end] = False
                task_inp[segment_begin:segment_end] = task
                assert segment_type[i] == 2
            segment_inp[segment_begin:segment_end] = segment_type[i]
            tgt[segment_begin : segment_end - 1] = np.where(
                context_inp[segment_begin + 1 : segment_end],
                -100,
                ctx[segment_begin + 1 : segment_end]
            )
            segment_begin = segment_end
        # prepend prompt segment
        context_inp = np.concatenate((np.full(self.prompt_length, True), context_inp))
        position_inp = np.concatenate((np.arange(self.prompt_length, dtype=np.int64), position_inp + self.prompt_length))
        segment_inp = np.concatenate((np.full(self.prompt_length, 0, dtype=np.int64), segment_inp))
        task_inp = np.concatenate((np.full(self.prompt_length, 0, dtype=np.int64), task_inp))
        tgt = np.concatenate((np.full(self.prompt_length, -100, dtype=np.int64), tgt))
        inp = np.concatenate((np.arange(self.prompt_length, dtype=np.int64) + self.prompt_length * global_task, ctx))
        return inp, tgt, inp.shape[0], context_inp, position_inp, segment_inp, task_inp

    def __getitem__(self, index):
        ctx = self.ctx[index]
        th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx = \
                self.__get_item_data(ctx, index)
        return th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx
