import time
from typing import Dict
import torch
import bmtrain as bmt
import json
import os
from cpm_live import get_args
import distutils.version  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

from cpm_live.models import CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMAntTokenizer
from training_tasks.bee import MixedDataset

def get_tokenizer(args):
    tokenizer = CPMAntTokenizer()
    return tokenizer


def get_model(args):
    config = CPMBeeConfig.from_json_file(args.model_config)
    model = CPMBee(config)
    if args.load is not None:
        bmt.load(model, args.load)
    else:
        bmt.init_parameters(model)
    return model


def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay, scale=args.loss_scale
    )
    if args.load is not None:
        if os.path.exists(os.path.join(args.save, args.save_name + (".rank-%d.opt" % 0))):
            # optimizer state exists
            states = torch.load(
                os.path.join(args.save, args.save_name + (".rank-%d.opt" % bmt.rank()))
            )
            optimizer.load_state_dict(states)
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    lr_scheduler = bmt.lr_scheduler.Noam(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup_iters,
        end_iter=args.lr_decay_iters,
        num_iter=args.start_step,
    )
    return lr_scheduler


def setup_model_and_optimizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler


def initialize():
    args = get_args()
    bmt.init_distributed(seed=args.seed, loss_scale_factor=2, loss_scale_steps=512)
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res


def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage


def pretrain(args, tokenizer, model, optimizer, lr_scheduler):

    average_time = bmt.utils.AverageRecorder()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step

    if bmt.rank() == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    global_token_pass = 0.0
    global_world_size = bmt.world_size()
    dataloader = MixedDataset(
        "datasets.json",
        args.batch_size,
        args.max_length,
        tokenizer,
        [0.5, 0.25, 0.25],
        max_depth=8
    )

    if os.path.exists(os.path.join(args.save, args.save_name + ("-%d.data.pt" % start_step))):
        # load dataset states if exists
        dataset_states = torch.load(
            os.path.join(args.save, args.save_name + ("-%d.data.pt" % start_step))
        )
        missing = dataloader.load_state_dict(dataset_states)
        if len(missing) > 0:
            bmt.print_rank("Missing keys when loading dataset states: ", missing)

    for iteration, data in enumerate(dataloader):

        iteration = iteration + start_step + 1
        assert len(data["ctx"]) == args.batch_size
        input_ids = torch.from_numpy(data["inputs"]).cuda().long()
        input_length = torch.from_numpy(data["length"]).cuda().long()
        input_context = torch.from_numpy(data["context"]).cuda().bool()
        input_sample_ids = torch.from_numpy(data["sample_ids"]).cuda().long()
        input_num_segments = torch.from_numpy(data["num_segments"]).cuda().long()
        input_segment_ids = torch.from_numpy(data["segment_ids"]).cuda().long()
        input_segment_rel_offset = torch.from_numpy(data["segment_rel_offset"]).cuda().long()
        input_segment_rel = torch.from_numpy(data["segment_rel"]).cuda().long()
        input_span = torch.from_numpy(data["spans"]).cuda().long()
        targets = torch.from_numpy(data["target"]).cuda().long()
        task_ids = torch.from_numpy(data["task_ids"]).cuda().long()
        task_names = data["task_names"]

        # ===========
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        mem_usage = {}
        tim_usage = {}
        mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

        # ===========
        logits, _ = model(
            input_ids,
            input_length,
            input_context,
            input_sample_ids,
            input_num_segments,
            input_segment_ids,
            input_segment_rel_offset,
            input_segment_rel,
            input_span
        )
        loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
        global_loss = bmt.sum_loss(loss).item()
        mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

        # ===========
        loss = optimizer.loss_scale(loss)
        loss.backward()
        mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

        # ===========
        grad_norm = bmt.optim.clip_grad_norm(
            optimizer.param_groups, args.clip_grad, scale=optimizer.scale, norm_type=2
        )
        bmt.optim_step(optimizer, lr_scheduler)
        mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)

        # ==========
        iteration_time = tim_usage["optim"] - tim_usage["init"]
        average_time.record(iteration_time)

        with torch.no_grad():
            task_num = len(task_names)
            logits_tmp : torch.Tensor = logits.view(1, -1, logits.size(-1)).expand(task_num, -1, -1)
            targets_tmp = targets.expand(task_num, -1, -1)
            
            task = torch.arange(task_num, dtype=torch.long, device="cuda")[:, None, None]
            targets_tmp = torch.where(task_ids == task, targets_tmp, -100)

            task_loss_map : Dict[str, float] = {}
            for i in range(task_num):
                task_loss = loss_func(logits_tmp[i, :], targets_tmp[i, :].view(-1))
                global_task_loss = float(bmt.sum_loss(task_loss).item())
                task_loss_map[task_names[i]] = global_task_loss

        local_total_rate = torch.Tensor([input_length.float().mean() / args.max_length]).cuda()
        local_total_rate = bmt.sum_loss(local_total_rate).item()
        global_token_pass += (
            global_world_size
            * local_total_rate
            * args.max_length
            * args.batch_size
        )
        avg_time = average_time.value

        train_info = {
            "time": tim_usage["init"],
            "iter": iteration,
            "loss": global_loss,
            "lr": lr_scheduler.current_lr,
            "lr scale": int(optimizer.scale),
            "time usage": tim_usage,
            "mem usage": mem_usage,
            "avg time (s)": avg_time,
            "token/max": local_total_rate,
            "token pass": global_token_pass,
            "throughout (token/s)": args.max_length * args.batch_size * local_total_rate / avg_time,
            "grad_norm": grad_norm.item(),
            "mask/max": ((targets >= 0).sum(-1).float().mean() / args.max_length).item(),
            "num_gpus": global_world_size,
        }
        train_info["task_loss"] = task_loss_map

        bmt.print_rank(
            (
                "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f} |"
                + " token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f}"
            ).format(
                iteration,
                global_loss,
                lr_scheduler.current_lr,
                int(optimizer.scale),
                avg_time,
                input_length.float().mean() / args.max_length,
                (targets >= 0).sum(-1).float().mean() / args.max_length,
                grad_norm,
            )
        )
        bmt.print_rank(
            "| "
            + " | ".join(
                [
                    "{} loss: {:.4f}".format(task_name, loss)
                    for task_name, loss in task_loss_map.items()
                ]
            )
        )
        if iteration % args.inspect_iters == 0:
            model_inspect = bmt.inspect.inspect_model(model, "*")
            bmt.print_rank(bmt.inspect.format_summary(model_inspect))
            train_info["model_inspect"] = model_inspect
        
        # write log here
        if bmt.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, iteration)
            for task_name, loss in task_loss_map.items():
                writer.add_scalar("Loss/train/{}".format(task_name), loss, iteration)

        if args.save is not None and iteration % args.save_iters == 0:
            bmt.save(model, os.path.join(args.save, args.save_name + ("-%d.pt" % iteration)))
            torch.save(
                optimizer.state_dict(),
                os.path.join(args.save, args.save_name + (".rank-%d.opt" % bmt.rank())),
            )
            all_states = dataloader.state_dict()
            if bmt.rank() == 0:
                # rank 0 writes the dataloader state
                torch.save(
                    all_states,
                    os.path.join(args.save, args.save_name + ("-%d.data.pt" % iteration)),
                )
            del all_states

    bmt.save(model, os.path.join(args.save, args.save_name + ".pt"))


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    pretrain(args, tokenizer, model, optimizer, lr_scheduler)


if __name__ == "__main__":
    main()
