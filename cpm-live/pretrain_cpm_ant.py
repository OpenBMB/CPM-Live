import time
from typing import Optional
import torch
import bmtrain as bmt
import json
import os
import datetime
from cpm_live.dataset import DistributedDataset
from cpm_live import get_args
import distutils.version  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

from cpm_live.models import CPMAnt, CPMAntConfig
from cpm_live.tokenizers import CPMAntTokenizer
from training_tasks.ant import CPMAntPretrainDataset


def get_tokenizer(args):
    tokenizer = CPMAntTokenizer()
    return tokenizer


def get_model(args):
    config = CPMAntConfig.from_json_file(args.model_config)
    model = CPMAnt(config)
    if args.load is not None:
        bmt.load(model, args.load)
    else:
        bmt.init_parameters(model)
    args.prompt_length = config.prompt_length
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


def get_log_time() -> datetime.datetime:
    return datetime.datetime.utcnow() + datetime.timedelta(hours=16)


def get_log_name(now: Optional[datetime.datetime] = None) -> str:
    if now is None:
        now = get_log_time()
    return "log.%s.txt" % now.strftime("%Y%m%d")


def lookup_latest_log():
    now = get_log_time()

    # try to find the latest log in the last 15 days
    for _ in range(15):
        log_name = get_log_name(now)
        if os.path.exists(log_name):
            with open(log_name, "r") as flog:
                line = flog.readlines()[-1]  # get last log
                return json.loads(line)

        now -= datetime.timedelta(days=1)  # try the previous day
    return None


def get_tasks():
    TASK_FILE_NAME = "tasks.json"
    if not hasattr(get_tasks, "info"):
        m_time = os.stat(TASK_FILE_NAME).st_mtime
        tasks = json.load(open(TASK_FILE_NAME, "r", encoding="utf-8"))
        get_tasks.info = {"m_time": m_time, "tasks": tasks}
    else:
        m_time = os.stat(TASK_FILE_NAME).st_mtime
        if m_time != get_tasks.info["m_time"]:
            tasks = json.load(open(TASK_FILE_NAME, "r", encoding="utf-8"))
            get_tasks.info = {"m_time": m_time, "tasks": tasks}
    return get_tasks.info["tasks"]


class BatchPacker:
    def __init__(self, dataset, max_length, batch_size):
        self.dataset = dataset
        self.max_length = max_length
        self.batch_size = batch_size

    def __iter__(self):
        ctx = []
        tgt = []
        context = []
        position = []
        segment = []
        span = []
        task_info = []

        for data in self.dataset:
            (
                ctx_data,
                tgt_data,
                _len,
                context_data,
                position_data,
                segment_data,
                task_data,
            ) = data
            if ctx_data is None:
                continue
            assert _len <= self.max_length

            ctx_data = ctx_data.astype("int64")
            tgt_data = tgt_data.astype("int64")

            for index in range(len(ctx)):
                if span[index][-1] + _len < self.max_length:
                    ctx[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        ctx_data
                    )[:_len].long()
                    tgt[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        tgt_data
                    )[:_len].long()
                    context[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        context_data
                    )[:_len].bool()
                    position[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        position_data
                    )[:_len].long()
                    segment[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        segment_data
                    )[:_len].long()
                    task_info[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        task_data
                    )[:_len].long()
                    span[index].append(span[index][-1] + _len)
                    break
            else:

                _ctx = torch.zeros((self.max_length,), dtype=torch.long)
                _ctx[:_len] = torch.from_numpy(ctx_data)[:_len].long()
                _tgt = torch.full((self.max_length,), -100, dtype=torch.long)
                _tgt[:_len] = torch.from_numpy(tgt_data)[:_len].long()
                _context = torch.full((self.max_length,), False, dtype=torch.bool)
                _context[:_len] = torch.from_numpy(context_data)[:_len].bool()
                _position = torch.full((self.max_length,), False, dtype=torch.long)
                _position[:_len] = torch.from_numpy(position_data)[:_len].long()
                _segment = torch.full((self.max_length,), False, dtype=torch.long)
                _segment[:_len] = torch.from_numpy(segment_data)[:_len].long()
                _task_info = torch.full((self.max_length,), -1, dtype=torch.long)
                _task_info[:_len] = torch.from_numpy(task_data)[:_len].long()
                ctx.append(_ctx)
                tgt.append(_tgt)
                context.append(_context)
                position.append(_position)
                segment.append(_segment)
                task_info.append(_task_info)
                span.append([_len])

            if len(ctx) > self.batch_size:
                _span = torch.zeros((self.batch_size, self.max_length + 1), dtype=torch.long)
                for bindex in range(self.batch_size):
                    for sindex in span[bindex]:
                        _span[bindex][sindex] = 1

                yield {
                    "ctx": torch.stack(ctx[: self.batch_size]),
                    "tgt": torch.stack(tgt[: self.batch_size]),
                    "context": torch.stack(context[: self.batch_size]),
                    "segment": torch.stack(segment[: self.batch_size]),
                    "position": torch.stack(position[: self.batch_size]),
                    "span": torch.cumsum(_span, dim=-1)[:, :-1],
                    "len_ctx": torch.LongTensor([it[-1] for it in span[: self.batch_size]]),
                    "task": torch.stack(task_info[: self.batch_size]),
                }

                ctx = ctx[self.batch_size :]
                tgt = tgt[self.batch_size :]
                context = context[self.batch_size :]
                segment = segment[self.batch_size :]
                position = position[self.batch_size :]
                span = span[self.batch_size :]
                task_info = task_info[self.batch_size :]


def pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset):

    average_time = bmt.utils.AverageRecorder()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step

    if bmt.rank() == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    global_token_pass = 0.0
    global_throughout = 0.0
    global_world_size = bmt.world_size()

    if bmt.rank() == 0:
        latest_log = lookup_latest_log()
        if latest_log is not None:
            global_token_pass = latest_log["token pass"]

    if os.path.exists(os.path.join(args.save, args.save_name + ("-%d.data.pt" % start_step))):
        # load dataset states if exists
        dataset_states = torch.load(
            os.path.join(args.save, args.save_name + ("-%d.data.pt" % start_step))
        )
        dataset.dataset.load_state_dict(dataset_states)

    dataloader = BatchPacker(dataset, args.max_length, args.batch_size)

    for iteration, data in enumerate(dataloader):

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
        mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

        # ===========
        logits, _ = model(
            input_idx,
            input_length,
            input_context,
            input_position,
            input_segment,
            input_span,
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

        task_ids = get_tasks()
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
                global_task_loss = bmt.gather_result(task_loss.unsqueeze(0)).nanmean().item()
                task_loss_list.append(global_task_loss)

        local_total_rate = torch.Tensor([input_length.float().mean() / args.max_length]).cuda()
        local_total_rate = bmt.sum_loss(local_total_rate).item()
        global_token_pass += (
            global_world_size
            * local_total_rate
            * (args.max_length - args.prompt_length)
            * args.batch_size
        )
        avg_time = average_time.value
        global_throughout += (
            (args.max_length - args.prompt_length) * args.batch_size * local_total_rate / avg_time
        )

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
            "throughout (token/s)": (args.max_length - args.prompt_length)
            * args.batch_size
            * local_total_rate
            / avg_time,
            "global throughout (token/s)": global_throughout / iteration,
            "grad_norm": grad_norm.item(),
            "mask/max": ((targets >= 0).sum(-1).float().mean() / args.max_length).item(),
            "num_gpus": global_world_size,
        }
        task_loss = {task_name: task_loss_list[idx] for (task_name, idx) in task_ids.items()}
        train_info["task_loss"] = task_loss

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
                    "{} loss: {:.4f}".format(task_name, task_loss_list[idx])
                    for task_name, idx in task_ids.items()
                ]
            )
        )
        if iteration % args.inspect_iters == 0:
            model_inspect = bmt.inspect.inspect_model(model, "*")
            bmt.print_rank(bmt.inspect.format_summary(model_inspect))
            train_info["model_inspect"] = model_inspect

        if bmt.rank() == 0:

            with open(get_log_name(), "a") as ff:
                ff.write(json.dumps(train_info) + "\n")

        if bmt.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, iteration)
            for i in task_ids.keys():
                writer.add_scalar("Loss/train/{}".format(i), task_loss_list[task_ids[i]], iteration)

        if args.save is not None and iteration % args.save_iters == 0:
            bmt.save(model, os.path.join(args.save, args.save_name + ("-%d.pt" % iteration)))
            torch.save(
                optimizer.state_dict(),
                os.path.join(args.save, args.save_name + (".rank-%d.opt" % bmt.rank())),
            )
            all_states = dataset.dataset.state_dict()
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
    dataset = CPMAntPretrainDataset(
        DistributedDataset("../data_bin_new", bmt.rank(), bmt.world_size()),
        max_length=args.max_length - args.prompt_length,
        prompt_length=args.prompt_length,
        tokenizer=tokenizer,
    )
    pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset)


if __name__ == "__main__":
    main()
