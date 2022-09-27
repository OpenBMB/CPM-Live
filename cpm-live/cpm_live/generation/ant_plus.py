import torch
from .ant import CPMAntGeneration, CPMAntBeamSearch, CPMAntRandomSampling


def convert_to_ids(tokenizer, text):
    ids = tokenizer.encode(text)
    ids = [j for j in ids if j != tokenizer.unk_id]
    return ids


class CPMAntPlusGeneration(CPMAntGeneration):
    def _convert_to_tensors(self, input_text, task_id=2):
        model_inputs = {}
        input_ids = [self.tokenizer.bos_id] + convert_to_ids(self.tokenizer, input_text)

        model_inputs["input"] = [
            x + self.prompt_length * task_id + self.tokenizer.vocab_size
            for x in range(self.prompt_length)
        ] + input_ids
        model_inputs["length"] = len(model_inputs["input"])
        model_inputs["position"] = list(range(len(model_inputs["input"])))
        model_inputs["span"] = [0] * len(model_inputs["input"])
        model_inputs["context"] = [True] * len(model_inputs["input"])
        model_inputs["segment"] = [0] * self.prompt_length + [2] * len(input_ids)

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs


class CPMAntPlusBeamSearch(CPMAntPlusGeneration, CPMAntBeamSearch):
    pass


class CPMAntPlusRandomSampling(CPMAntPlusGeneration, CPMAntRandomSampling):
    pass


class CPMAntPlusQuestionAnswering(CPMAntPlusBeamSearch):
    def _convert_to_tensors(self, text_list, task_id=4):
        model_inputs = {}
        doc = text_list[0]
        question_list = text_list[1:]
        # doc ids
        doc_ids = (
            [self.tokenizer.bos_id] + convert_to_ids(self.tokenizer, doc) + [self.tokenizer.eos_id]
        )
        # question ids
        sep_id = 3
        question_ids = [self.tokenizer.bos_id]
        for idx, q in enumerate(question_list):
            question_ids += convert_to_ids(self.tokenizer, q)
            if idx != len(question_list) - 1:
                question_ids.append(sep_id)
        question_ids += [self.tokenizer.eos_id]
        # bos_id for answers
        answer_ids = [self.tokenizer.bos_id]

        model_inputs["input"] = (
            [
                x + self.prompt_length * task_id + self.tokenizer.vocab_size
                for x in range(self.prompt_length)
            ]
            + doc_ids
            + question_ids
            + answer_ids
        )
        model_inputs["length"] = len(model_inputs["input"])
        model_inputs["position"] = list(range(len(model_inputs["input"])))
        model_inputs["span"] = [0] * len(model_inputs["input"])
        model_inputs["context"] = [True] * len(model_inputs["input"])
        model_inputs["segment"] = (
            [0] * self.prompt_length
            + [1] * len(doc_ids)
            + [2] * len(question_ids)
            + [3] * len(answer_ids)
        )

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs


class CPMAntPlusSummarization(CPMAntPlusBeamSearch):
    def _convert_to_tensors(self, input_text, task_id=3):
        model_inputs = {}
        # doc ids
        doc_ids = (
            [self.tokenizer.bos_id]
            + convert_to_ids(self.tokenizer, input_text)
            + [self.tokenizer.eos_id]
        )
        # bos_id for answers
        answer_ids = [self.tokenizer.bos_id]

        model_inputs["input"] = (
            [
                x + self.prompt_length * task_id + self.tokenizer.vocab_size
                for x in range(self.prompt_length)
            ]
            + doc_ids
            + answer_ids
        )
        model_inputs["length"] = len(model_inputs["input"])
        model_inputs["position"] = list(range(len(model_inputs["input"])))
        model_inputs["span"] = [0] * len(model_inputs["input"])
        model_inputs["context"] = [True] * len(model_inputs["input"])
        model_inputs["segment"] = (
            [0] * self.prompt_length + [1] * len(doc_ids) + [2] * len(answer_ids)
        )

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs
