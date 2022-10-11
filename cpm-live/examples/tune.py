import os
import sys
import time
import torch
import numpy as np
import bmtrain as bmt
import distutils.version  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, "..")
from cpm_live.utils import pad
from dataset import JsonlDataset, DistributedDataLoader


def pad_collate_fn():
    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    "The elements of the batch contain different keys."
                    f"Cannot batch them ({set(item.keys())} != {keys})"
                )
        padded = {}
        for key in keys:
            if key == "target":
                _padding_value = -100
            else:
                _padding_value = 0
            padded[key] = pad(items, key, _padding_value, padding_side="right")
        return padded

    return inner


class CPMAntPlusTune:
    def __init__(
        self,
        model,
        tokenizer,
        prompt_length=32,
        lr=5e-3,
        warmup_iters=50,
        task_id=2,
        max_len=256,
        cls_num=None,
        epochs=1,
        batch_size=1,
        num_workers=1,
        eval_interval=50,
        output_path="output",
        early_stop_patience=None,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = bmt.optim.AdamOffloadOptimizer(
            model.parameters(), weight_decay=0.01, scale=1048576
        )
        self.lr_scheduler = bmt.lr_scheduler.Noam(
            self.optimizer, start_lr=lr, warmup_iter=warmup_iters, end_iter=-1, num_iter=0
        )
        self.loss_function = bmt.loss.FusedCrossEntropy(ignore_index=-100)
        self.task_id = task_id
        self.max_len = max_len
        self.prompt_length = prompt_length
        self.cls_num = cls_num
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.num_workers = num_workers
        self.batch_size = batch_size
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.early_stop_patience = early_stop_patience

        tensorboard_log_path = os.path.join(output_path, "logs")
        if bmt.rank() == 0 and tensorboard_log_path is not None:
            self.summary_writer = SummaryWriter(log_dir=tensorboard_log_path)
        else:
            self.summary_writer = None

    def _ensure_tensor_on_device(self, inputs, device):

        if isinstance(inputs, dict):
            return {
                name: self._ensure_tensor_on_device(tensor, device)
                for name, tensor in inputs.items()
            }
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype == torch.float16:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def process_data(self, inputs, **kwargs):
        raise NotImplementedError("process_data is not implemented")

    def _forward(self, inputs, **kwargs):
        raise NotImplementedError("_forward is not implemented")

    def forward(self, train_dataloader, eval_dataloader, cls_num=None):

        average_time = 0
        average_time_shift = 0.9
        global_step = 0
        best_eval_loss = 1e9
        best_eval_step = 0

        self.optimizer.zero_grad()
        for epoch in range(self.epochs):
            for idx, train_data in enumerate(train_dataloader):
                train_data = self._ensure_tensor_on_device(train_data, device="cuda")
                self.model.train()
                global_step += 1

                start_time = time.time()
                # custom part for different models

                loss = self._forward(train_data, cls_num=cls_num)
                global_loss = bmt.sum_loss(loss).item()
                loss = self.optimizer.loss_scale(loss)
                loss.backward()
                grad_norm = bmt.optim.clip_grad_norm(
                    self.optimizer.param_groups,
                    max_norm=10.0,
                    scale=self.optimizer.scale,
                    norm_type=2,
                )
                bmt.optim_step(self.optimizer, self.lr_scheduler)

                iteration_time = time.time() - start_time
                average_time = (
                    average_time * average_time_shift + (1 - average_time_shift) * iteration_time
                )

                bmt.print_rank(
                    "| Train | Epoch: {:3d} | Iter: {:6d} | loss: {:.4f} |"
                    "lr: {:.4e}, scale: {:10.4f} | time: {:.4f} | grad_norm: {:.4f}".format(
                        epoch,
                        idx,
                        global_loss,
                        self.lr_scheduler.current_lr,
                        int(self.optimizer.scale),
                        average_time / (1 - pow(average_time_shift, global_step + 1)),
                        grad_norm,
                    )
                )

                if bmt.rank() == 0 and self.summary_writer is not None:
                    self.summary_writer.add_scalar("Loss/train", global_loss, global_step)

                self.optimizer.zero_grad()

                if global_step % self.eval_interval == 0:
                    self.model.eval()

                    total_loss = 0
                    cnt = 0
                    with torch.inference_mode():
                        for eval_data in eval_dataloader:
                            cnt += 1
                            eval_data = self._ensure_tensor_on_device(eval_data, device="cuda")
                            loss = self._forward(eval_data, cls_num=cls_num)
                            total_loss += bmt.sum_loss(loss).item()

                    assert cnt == len(eval_dataloader)
                    eval_loss = total_loss / cnt

                    if bmt.rank() == 0 and self.summary_writer is not None:
                        self.summary_writer.add_scalar("Loss/eval", eval_loss, global_step)

                    bmt.print_rank(
                        "| Eval | Iter: {:6d} | loss: {:.4f}".format(global_step, eval_loss)
                    )

                    # save best model
                    if eval_loss < best_eval_loss:
                        bmt.print_rank(
                            "[INFO] Iteration {} is the best checkpoint now!".format(global_step)
                        )
                        best_eval_loss = eval_loss
                        best_eval_step = global_step
                        ckpt_full_path = os.path.join(self.output_path, "best.pt")

                        state_dict = self.model.state_dict()
                        if bmt.rank() == 0:
                            torch.save(state_dict, ckpt_full_path)
                    elif (
                        self.early_stop_patience is not None
                        and (global_step - best_eval_step) // self.eval_interval
                        >= self.early_stop_patience
                    ):
                        bmt.print_rank("[INFO] Early stop at iteration {}!".format(global_step))
                        return
            bmt.print_rank(f"[INFO] Epoch {epoch} finished!")
        return

    def run(self, inputs):
        collate_fn = pad_collate_fn()

        train_dataset = JsonlDataset(inputs["train"], self.process_data)
        train_dataloader = DistributedDataLoader(
            train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

        eval_dataset = JsonlDataset(inputs["eval"], self.process_data)
        eval_dataloader = DistributedDataLoader(
            eval_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

        self.forward(train_dataloader, eval_dataloader, cls_num=self.cls_num)


class CPMAntPlusNLGTune(CPMAntPlusTune):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.truncate_num = 0

    def process_data(self, inputs):
        res = {}
        target = inputs["target"]
        input_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(inputs["input"])
        target_ids = self.tokenizer.encode(target) + [self.tokenizer.eos_id]

        if self.prompt_length + len(input_ids) + len(target_ids) > self.max_len:
            self.truncate_num += 1
            if self.truncate_num % 100 == 0:
                bmt.print_rank(
                    f"[Warning] There are more than {self.truncate_num} instances are truncated!"
                    "Consider to increase max_len!"
                )

            tr_input_length = self.max_len - self.prompt_length - len(target_ids)
            if tr_input_length > 0:
                input_ids = input_ids[-tr_input_length:]
            else:
                # target is too long
                bmt.print_rank(
                    f"[Warning] target {target} length exceeds max_len, check your data!"
                )
                input_ids = []
                target_ids = target_ids[-(self.max_len - self.prompt_length) :]

        res["input"] = (
            [x + self.prompt_length * self.task_id + self.tokenizer.vocab_size for x in range(self.prompt_length)]
            + input_ids
            + target_ids
        )
        assert len(res["input"]) <= self.max_len
        res["length"] = len(res["input"])
        res["position"] = list(range(len(res["input"])))
        res["span"] = [0] * len(res["input"])
        res["context"] = [True] * (len(res["input"]) - len(target_ids)) + [False] * len(target_ids)

        res["segment"] = [0] * self.prompt_length
        res["segment"] += [2] * len(input_ids)
        res["segment"] += [2] * len(target_ids)

        tgt = np.full(len(res["input"]), -100, dtype=np.int32)
        tgt[:-1] = np.where(res["context"][1:], -100, res["input"][1:])
        res["target"] = tgt

        for key in res:
            res[key] = torch.tensor(res[key]).int().unsqueeze(0)

        return res

    def _forward(self, model_inputs, **kwargs):
        targets = model_inputs.pop("target", None)
        logits, _ = self.model(**model_inputs)
        loss = self.loss_function(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return loss


class CPMAntPlusNLUTune(CPMAntPlusTune):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.truncate_num = 0

    def process_data(self, inputs):
        target = inputs["target"]
        assert isinstance(target, int), "target must be an int in nlu tasks!"
        option_list = inputs["options"]

        input_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(inputs["input"])

        res = {}
        res["input"] = []
        res["length"] = []
        res["position"] = []
        res["span"] = []
        res["context"] = []
        res["segment"] = []

        for option in option_list:
            cur_input_ids = (
                input_ids + self.tokenizer.encode(option) + self.tokenizer.encode("[是否正确]")
            )
            if self.prompt_length + len(cur_input_ids) > self.max_len:
                self.truncate_num += 1
                if self.truncate_num % 100 == 0:
                    bmt.print_rank(
                        f"There are more than {self.truncate_num} instances are truncated!"
                        "Consider to increse max_len!"
                    )

                tr_input_length = self.max_len - self.prompt_length
                assert tr_input_length > 0
                cur_input_ids = cur_input_ids[-tr_input_length:]

            ids = [
                x + self.prompt_length * self.task_id + self.tokenizer.vocab_size for x in range(self.prompt_length)
            ] + cur_input_ids
            res["input"].append(ids)
            res["length"].append(len(ids))
            res["position"].append(list(range(len(ids))))
            res["span"].append([0] * len(ids))
            res["context"].append([True] * len(ids))
            res["segment"].append([0] * self.prompt_length + [2] * len(cur_input_ids))

        for key in res:
            for i in range(len(res[key])):
                res[key][i] = torch.tensor(res[key][i]).int().unsqueeze(0)

        res["target"] = torch.tensor(target).int().unsqueeze(0)

        return res

    def _forward(self, model_inputs, cls_num):
        targets = model_inputs.pop("target", None)
        output, _ = self.model(**model_inputs)
        output = output[:, :, self.tokenizer.encode("是")[0]]
        logits = (
            output[torch.arange(output.size(0)), (model_inputs["length"] - 1).long()]
            .unsqueeze(1)
            .view(-1, cls_num)
        )
        loss = self.loss_function(logits, targets.view(-1))

        return loss
