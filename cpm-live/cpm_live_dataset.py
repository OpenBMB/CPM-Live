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
        n_segment = raw_data[0]
        len_info = n_segment * 2 + 1
        segment_len = raw_data[1:len_info:2] 
        segment_type = raw_data[2:len_info:2]
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
            
            # random select a task
            task_p = random.random()
            if task_p <= 0.5:
                # multi span
                task = 0
            else:
                # prefix lm
                task = 1

            # generate target
            if task == 0:
                num_mask = random.randint(1, segment_len[i] - 1)
                mask_idx = np.random.choice(segment_len[i] - 1, num_mask, replace=False) + segment_begin
                context_inp[mask_idx + 1] = False
                task_inp[segment_begin:segment_end] = task
            elif task == 1:
                num_mask = random.randint(1, segment_len[i] - 1)
                context_inp[segment_end - num_mask:segment_end] = False
                task_inp[segment_begin:segment_end] = task
            
            segment_inp[segment_begin:segment_end] = segment_type[i]

            segment_begin = segment_end
            
        tgt[:-1] = np.where(
            context_inp[1:],
            -100,
            inp[1:]
        )

        # prepend prompt segment
        context_inp = np.concatenate((np.full(self.prompt_length, True), context_inp))
        position_inp = np.concatenate((np.arange(self.prompt_length, dtype=np.int64), position_inp + self.prompt_length))
        segment_inp = np.concatenate((np.full(self.prompt_length, 0, dtype=np.int64), segment_inp))
        task_inp = np.concatenate((np.full(self.prompt_length, 0, dtype=np.int64), task_inp))
        tgt = np.concatenate((np.full(self.prompt_length, -100, dtype=np.int64), tgt))
        inp = np.concatenate((np.arange(self.prompt_length, dtype=np.int64), ctx))

        return inp, tgt, inp.shape[0], context_inp, position_inp, segment_inp, task_inp

    def __getitem__(self, index):
        ctx = self.ctx[index]
        th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx = \
                self.__get_item_data(ctx, index)
        return th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx
