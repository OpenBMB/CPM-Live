import torch
import sys

sys.path.insert(0, "..")
from cpm_live.generation import CPMAntGeneration, CPMAntBeamSearch


class CPMAntNLGInfer(CPMAntBeamSearch):
    def _convert_to_tensors(self, inputs, task_id=1):
        input_text = inputs["input"]
        return super()._convert_to_tensors(input_text, task_id)


class CPMAntNLUInfer(CPMAntGeneration):
    def _convert_to_tensors(self, inputs, task_id=1):
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
            ids = (
                [x + self.prompt_length * task_id for x in range(self.prompt_length)]
                + input_ids
                + self.tokenizer.encode(option)
                + self.tokenizer.encode("[是否正确]")
            )
            res["input"].append(ids)
            res["length"].append(len(ids))
            res["context"].append([True] * len(ids))
            res["position"].append(list(range(len(ids))))
            res["segment"].append([0] * self.prompt_length + [2] * (len(ids) - self.prompt_length))
            res["span"].append([0] * len(ids))

        for key in res:
            for i in range(len(res[key])):
                res[key][i] = torch.tensor(res[key][i]).int().unsqueeze(0)

        return res

    def _decode(self, model_inputs, cls_num, **kwargs):
        output, _, _ = self.model.inference(**model_inputs)
        logits = output[:, -1, self.tokenizer.encode("是")].view(-1, cls_num)
        result = torch.argmax(logits, -1)
        return result.cpu().tolist()


class CPMAntScoreInfer(CPMAntGeneration):
    def _convert_to_tensors(self, inputs, task_id=1):
        input_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(inputs["input"])

        res = {}
        ids = (
            [x + self.prompt_length * task_id for x in range(self.prompt_length)]
            + input_ids
            + self.tokenizer.encode("[是否正确]")
        )
        res["input"] = ids
        res["length"] = len(ids)
        res["context"] = [True] * len(ids)
        res["position"] = list(range(len(ids)))
        res["segment"] = [0] * self.prompt_length + [2] * (len(ids) - self.prompt_length)
        res["span"] = [0] * len(ids)

        for key in res:
            res[key] = torch.tensor(res[key]).int().unsqueeze(0)

        return res

    def _decode(self, model_inputs, **kwargs):
        output, _, _ = self.model.inference(**model_inputs)
        scores = output[:, -1, self.tokenizer.encode("是")[0]]
        return scores.cpu().tolist()
