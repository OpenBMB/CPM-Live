import time
import random
import torch
import bmtrain as bmp
import json
from bmtrain import nccl
from bmtrain.global_var import config
import numpy as np
import os

from model_center.dataset import DistributedMMapIndexedDataset, MMapIndexedDataset
from model_center.arguments import get_args
import distutils.version
from torch.utils.tensorboard import SummaryWriter

from cpm_live_config import CPMLiveConfig
from cpm_live_model import CPMLive
from cpm_live_tokenizer import CPMLiveTokenizer
from cpm_live_dataset import CPMLive_Dataset

def get_tokenizer(args):
    tokenizer = CPMLiveTokenizer(args.vocab_file)
    return tokenizer

def get_model(args):
    config = CPMLiveConfig.from_json_file(args.model_config)
    model = CPMLive(config)
    if args.load != None:
        bmp.load(model, args.load)
    else:
        bmp.init_parameters(model)
    args.prompt_length = config.prompt_length
    return model

def get_optimizer(args, model):
    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    lr_scheduler = bmp.lr_scheduler.Noam(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step)
    return lr_scheduler

def setup_model_and_optimizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    bmp.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmp.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    args = get_args()
    bmp.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2), 
         round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2))
    torch.cuda.reset_peak_memory_stats()
    return res

def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage

def batch_iter(args, dataset, start_step = 0):
    st = 0
    ctx = []
    tgt = []
    context = []
    position = []
    segment = []
    span = []
    task_info = []

    exist_total = 0
    while True:
        ctx_data, tgt_data, _len, context_data, position_data, segment_data, task_data = dataset[st]
        st += 1
        if ctx_data is None:
            continue
        assert _len <= args.max_length

        ctx_data = ctx_data.astype("int64")
        tgt_data = tgt_data.astype("int64")

        for index in range(len(ctx)):
            if span[index][-1] + _len < args.max_length:
                ctx[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(ctx_data)[:_len].long()
                tgt[index][span[index][-1]:span[index][-1] + _len]= torch.from_numpy(tgt_data)[:_len].long()
                context[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(context_data)[:_len].bool()
                position[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(position_data)[:_len].float()
                segment[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(segment_data)[:_len].long()
                task_info[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(task_data)[:_len].long()
                span[index].append(span[index][-1] + _len)
                break
        else:
            _ctx = torch.zeros((args.max_length,), dtype=torch.long)
            _ctx[:_len] = torch.from_numpy(ctx_data)[:_len].long()
            _tgt = torch.full((args.max_length,), -100, dtype=torch.long)
            _tgt[:_len] = torch.from_numpy(tgt_data)[:_len].long()
            _context = torch.full((args.max_length,), False, dtype=torch.bool)
            _context[:_len] = torch.from_numpy(context_data)[:_len].bool()
            _position = torch.full((args.max_length,), False, dtype=torch.float)
            _position[:_len] = torch.from_numpy(position_data)[:_len].float()
            _segment = torch.full((args.max_length,), False, dtype=torch.long)
            _segment[:_len] = torch.from_numpy(segment_data)[:_len].long()
            _task_info = torch.full((args.max_length,), -1, dtype=torch.long)
            _task_info[:_len] = torch.from_numpy(task_data)[:_len].long()
            ctx.append(_ctx)
            tgt.append(_tgt)
            context.append(_context)
            position.append(_position)
            segment.append(_segment)
            task_info.append(_task_info)
            span.append([_len])

        if len(ctx) > args.batch_size:
            if exist_total >= start_step:
                _span = torch.zeros((args.batch_size, args.max_length + 1), dtype=torch.long)
                for bindex in range(args.batch_size):
                    for sindex in span[bindex]:
                        _span[bindex][sindex] = 1
                
                yield {
                    "ctx": torch.stack(ctx[:args.batch_size]),
                    "tgt": torch.stack(tgt[:args.batch_size]),
                    "context": torch.stack(context[:args.batch_size]),
                    "segment": torch.stack(segment[:args.batch_size]),
                    "position": torch.stack(position[:args.batch_size]),
                    "span": torch.cumsum(_span, dim=-1)[:,:-1],
                    "len_ctx": torch.LongTensor([it[-1] for it in span[:args.batch_size]]),
                    "task": torch.stack(task_info[:args.batch_size]),
                }
            exist_total += 1
            ctx = ctx[args.batch_size:]
            tgt = tgt[args.batch_size:]
            context = context[args.batch_size:]
            segment = segment[args.batch_size:]
            position = position[args.batch_size:]
            span = span[args.batch_size:]
            task_info = task_info[args.batch_size:]

def pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset):

    average_time = 0
    average_time_shift = 0.9
    loss_func = bmp.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step
    task_ids = {"mlm": 0, "lm": 1}

    if bmp.rank() == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    global_token_pass = 0.0
    global_throughout = 0.0
    global_word_size = bmp.world_size()

    for iteration, data in enumerate(batch_iter(args, dataset, start_step)):

        iteration = iteration + start_step + 1
        assert len(data["ctx"]) == args.batch_size
        input_idx = data["ctx"].int().cuda()
        input_length = data["len_ctx"].int().cuda()
        input_context = data["context"].bool().cuda()
        input_position = data["position"].float().cuda()
        input_segment = data["segment"].int().cuda()
        input_span = data["span"].int().cuda()
        targets = data["tgt"].long().cuda()
        task_info = data["task"].long().cuda()
        
        # ===========
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        mem_usage = {}
        tim_usage = {}
        mem_usage, tim_usage = add_mem_time('init', mem_usage, tim_usage)

        # ===========
        logits, _ = model(input_idx, input_length, input_context, input_position, input_segment, input_span)
        loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
        global_loss = bmp.sum_loss(loss).item()
        mem_usage, tim_usage = add_mem_time('forward', mem_usage, tim_usage)

        # ===========
        loss = optimizer.loss_scale(loss)
        loss.backward()
        mem_usage, tim_usage = add_mem_time('backward', mem_usage, tim_usage)
        
        # ===========
        grad_norm = bmp.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)    
        bmp.optim_step(optimizer, lr_scheduler)
        mem_usage, tim_usage = add_mem_time('optim', mem_usage, tim_usage)

        # ==========s
        iteration_time = tim_usage['optim'] - tim_usage['init']
        average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time

        with torch.no_grad():
            task_num = len(task_ids)
            logits_tmp = logits.view(-1, logits.size(-1)).expand(task_num, -1, -1)
            targets_tmp = targets.expand(task_num, -1, -1)
            task_info = task_info.expand(task_num, -1, -1)

            task = task_info.new([x for x in range(task_num)])[:, None, None]
            targets_tmp = torch.where(task_info == task, targets_tmp, -100)

            task_loss_list = []
            for i in range(task_num):
                task_loss = loss_func(logits_tmp[i, :], targets_tmp[i, :].view(-1))
                global_task_loss = bmp.gather_result(task_loss.unsqueeze(0)).nanmean().item()
                task_loss_list.append(global_task_loss)

        local_total_rate = torch.Tensor([input_length.float().mean() / args.max_length]).cuda()
        local_total_rate = bmp.sum_loss(local_total_rate).item()
        global_token_pass += (global_word_size * local_total_rate * (args.max_length - 32) * args.batch_size) 
        avg_time = (average_time / (1 - pow(average_time_shift, iteration + 1)))
        global_throughout += (args.max_length - args.prompt_length) * args.batch_size * local_total_rate / avg_time

        train_info = {
            'time': tim_usage['init'],
            'iter': iteration,
            'loss': global_loss,
            'lr': lr_scheduler.current_lr,
            'lr scale': int(optimizer.scale),
            'time usage': tim_usage,
            'mem usage': mem_usage,
            'avg time (s)': avg_time,
            'token/max': local_total_rate,
            'token pass': global_token_pass,
            'throughout (token/s)': (args.max_length - args.prompt_length) * args.batch_size * local_total_rate / avg_time,
            'global throughout (token/s)': global_throughout / iteration,
            'grad_norm': grad_norm.item(),
            'mask/max': ((targets>=0).sum(-1).float().mean()/args.max_length).item(),
        }
        task_loss = {task_name: task_loss_list[idx] for (task_name, idx) in task_ids.items()}
        train_info['task_loss'] = task_loss

        bmp.print_rank(
            "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f} | token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f}".format(
                iteration,
                global_loss,
                lr_scheduler.current_lr,
                int(optimizer.scale),
                average_time / (1 - pow(average_time_shift, iteration + 1)),
                input_length.float().mean()/args.max_length,
                (targets>=0).sum(-1).float().mean()/args.max_length,
                grad_norm
            )
        )
        bmp.print_rank(
            "| " + " | ".join(["{} loss: {:.4f}".format(task_name, task_loss_list[idx]) for task_name, idx in task_ids.items()])
        )
        if iteration % args.inspect_iters == 0 and bmp.rank() == 0:
            model_inspect = bmp.inspect.inspect_model(model, "*")
            bmp.print_rank(
                bmp.inspect.format_summary(
                    model_inspect
                )
            )
            train_info['model_inspect'] = model_inspect

        if bmp.rank() == 0:
            ff = open("log.txt", "a")
            ff.write(json.dumps(train_info)+"\n")
            ff.close()

        if bmp.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, iteration)
            for i in task_ids.keys():
                writer.add_scalar("Loss/train/{}".format(i), task_loss_list[task_ids[i]], iteration)
        
        if args.save != None and iteration % args.save_iters == 0:
            bmp.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % iteration)))


    bmp.save(model, os.path.join(args.save, args.save_name+".pt"))


def main():
    os.environ["MASTER_PORT"]=  (str)((int)(os.environ["MASTER_PORT"]) + 1234)
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = CPMLive_Dataset(
        DistributedMMapIndexedDataset("../data_bin/", "cpm_live_text_context", bmp.rank(), bmp.world_size()),
        max_length = args.max_length - args.prompt_length, 
        prompt_length = args.prompt_length,
        tokenizer = tokenizer
    )
    pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
