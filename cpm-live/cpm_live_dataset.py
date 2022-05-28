import torch
import torch.utils.data as data
import random
import numpy as np

import torch
import numpy as np
import random
import torch.nn.functional as F

class CPMLive_Dataset(data.Dataset):
    def __init__(self, ctx, max_length = 1024, prompt_length = 32, tokenizer = None):
        self.ctx = ctx
        self.max_length = max_length + prompt_length
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ctx)

    def __get_item_data(self, ctx, index):
        # 超长就跳过该条数据
        if ctx.shape[0] > self.max_length - self.prompt_length:
            return None, None, None, None, None, None, None

        task = random.random()
        if task <= 0.4:
            task = 1 # multi span
        elif task <= 0.8:
            task = 2 # prefix single span
        else:
            task = 3 # prefix single span

        len_ctx = min(ctx.shape[0], self.max_length - self.prompt_length)
        inp = np.arange((self.prompt_length + len_ctx), dtype = np.int64) + \
             self.prompt_length * task
        inp[self.prompt_length:] = ctx[:len_ctx]
        len_inp = len(inp)

        context_inp = np.full(len_inp, True)
        position_inp = np.arange((len_inp), dtype = np.float32)
        segment_inp = np.zeros((len_inp), dtype = np.int64)

        if task == 1:
            num_mask = random.randint(1, len_ctx - 1)
            mask_idx = np.random.choice(len_ctx - 1, num_mask, replace=False)
            context_inp[mask_idx + 1 + self.prompt_length] = False
            segment_inp[self.prompt_length:] = 1
            task_inp = np.full((len_inp), 0, dtype = np.int64)
        elif task == 2:
            num_mask = random.randint(1, len_ctx - 1)
            if random.random() < 0.5:
                num_mask = len_ctx - 1
            context_inp[-num_mask:] = False
            segment_inp[self.prompt_length:] = 2
            task_inp = np.full((len_inp), 1, dtype = np.int64)
        elif task == 3:
            num_mask = random.randint((len_ctx - 1) // 4, (len_ctx - 1) // 2)
            context_inp[-num_mask:] = False
            segment_inp[-num_mask:] = 2

            num_mask_mlm = random.randint(1, len_ctx - 1 - num_mask)
            mask_idx = np.random.choice(len_ctx - 1 - num_mask, num_mask_mlm, replace=False)
            context_inp[mask_idx + 1 + self.prompt_length] = False
            segment_inp[self.prompt_length : -num_mask] = 1

            task_inp = np.full((len_inp), 0, dtype = np.int64)
            task_inp[-num_mask:] = 1

        tgt = np.full((len_inp), -100, dtype = np.int64)
        tgt[:-1] = np.where(
            context_inp[1:],
            -100,
            inp[1:]
        )

        if tgt[-2] >= 0 and random.random() < 0.5:
            tgt[-2] = -100

        # for index in range(len(tgt)):
        #     if tgt[index] != -100:
        #         assert index >= self.prompt_length
        #         assert tgt[index] == inp[index+1]

        return inp, tgt, len_inp, context_inp, position_inp, segment_inp, task_inp

    def __getitem__(self, index):
        ctx = self.ctx[index]
        th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx = \
                self.__get_item_data(ctx, index)
        return th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx
