import torch

class BeamHypotheses(object):
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty

        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / cur_len**self.length_penalty


def _repetition_penalty(
    logits,
    batch_size,
    num_beams,
    prev_output_tokens,
    repetition_penalty,
    start_idx=None,
    end_idx=None,
    window_size=None,
):
    # only conduct repetition penaly for the output
    assert repetition_penalty >= 1, "repetition penalty coefficient should >= 1"
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    for i in range(batch_size * num_beams):
        if start_idx is None or end_idx is None:
            output_tokens = prev_output_tokens[i].tolist()
        else:
            if end_idx >= start_idx:
                if window_size:
                    output_tokens = prev_output_tokens[i][
                        max(start_idx, end_idx + 1 - window_size) : end_idx + 1
                    ].tolist()
                else:
                    output_tokens = prev_output_tokens[i][start_idx : end_idx + 1].tolist()
            else:
                output_tokens = []
        for previous_token in set(output_tokens):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if logits[i, previous_token] < 0:
                logits[i, previous_token] *= repetition_penalty
            else:
                logits[i, previous_token] /= repetition_penalty


def beam_search(
    self, model_inputs, beam_size, generate_length, repetition_penalty=1.0, repetition_window=None
):
    """
    Beam search
    Args:
        model_inputs (dict): input ids
        beam_size (int): beam size of beam search
        generate_length (int): maximum generation length
        repetition_penalty (float, optional): repetition penalty coefficient, 1.0 means no penalty
        repetition_window (int, optional): window size of repetition penalty, None means that all output tokens are penalized.
    """
    # generate_length + 1 for EOS token
    generate_length += 1

    # expand dimmension
    batch_size = model_inputs["input"].size(0)
    input = (
        model_inputs["input"]
        .unsqueeze(1)
        .expand(batch_size, beam_size, -1)
        .contiguous()
        .view(batch_size * beam_size, -1)
    )
    length = (
        model_inputs["length"]
        .unsqueeze(1)
        .expand(batch_size, beam_size)
        .contiguous()
        .view(
            batch_size * beam_size,
        )
    )
    context = (
        model_inputs["context"]
        .unsqueeze(1)
        .expand(batch_size, beam_size, -1)
        .contiguous()
        .view(batch_size * beam_size, -1)
    )
    position = (
        model_inputs["position"]
        .unsqueeze(1)
        .expand(batch_size, beam_size, -1)
        .contiguous()
        .view(batch_size * beam_size, -1)
    )
    segment = (
        model_inputs["segment"]
        .unsqueeze(1)
        .expand(batch_size, beam_size, -1)
        .contiguous()
        .view(batch_size * beam_size, -1)
    )
    span = (
        model_inputs["span"]
        .unsqueeze(1)
        .expand(batch_size, beam_size, -1)
        .contiguous()
        .view(batch_size * beam_size, -1)
    )

    done = [False for _ in range(batch_size)]

    beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(beam_size, generate_length, length_penalty=1, early_stopping=False)
        for _ in range(batch_size)
    ]

    pred_start_index = input.size(-1)
    past_key_values = None
    for i in range(generate_length + 1):
        if i == 0:
            logits, _, past_key_values = self.model(
                input=input,
                length=length,
                context=context,
                position=position,
                segment=segment,
                span=span,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            logits, _, past_key_values = self.model(
                input=input[:, -1:],
                length=length,
                context=context,
                position=position,
                segment=segment,
                span=span,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # skip all steps when we are done with each sentence
        if all(done):
            break  # Note: break not supports multi-GPUs

        # (batch * beam, seqlen, model_dim)
        logits = logits[:, -1, :]

        _repetition_penalty(
            logits,
            batch_size,
            beam_size,
            input,
            repetition_penalty,
            pred_start_index,
            input.size(-1) - 1,
            repetition_window,
        )
        scores = torch.nn.functional.log_softmax(logits, dim=-1)

        next_scores = scores + beam_scores[:, None].expand_as(
            scores
        )  # (batch_size * beam_size, vocab_size)

        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
        next_scores, next_words = torch.topk(
            next_scores, 2 * beam_size, dim=1, largest=True, sorted=True
        )

        assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)
        next_batch_beam = []

        for sent_id in range(batch_size):
            # if we are done with this sentence
            done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                next_scores[sent_id].max().item(), i
            )
            if done[sent_id]:
                next_batch_beam.extend([(0, self.tokenizer.pad_id, 0)] * beam_size)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                # get beam and word IDs
                beam_id = torch.div(idx, scores.size(-1), rounding_mode="floor")
                word_id = idx % scores.size(-1)

                # end of sentence, or next word
                if word_id == self.tokenizer.eos_id or i == generate_length:
                    if i > 0:
                        generated_hyps[sent_id].add(
                            input[sent_id * beam_size + beam_id, pred_start_index:].clone(),
                            value.item(),
                        )
                else:
                    next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == beam_size:
                    break

            # update next beam content
            assert len(next_sent_beam) == 0 if i == generate_length else beam_size
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, self.tokenizer.pad_id, 0)] * beam_size  # pad the batch
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == beam_size * (sent_id + 1)

        # At the last step, we should not add the token to the next position
        if i == generate_length:
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * beam_size
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_words = input.new([x[1] for x in next_batch_beam])
        beam_idx = length.new([x[2] for x in next_batch_beam]).long()

        # re-order batch and internal states
        input = input[beam_idx, :]

        past_key_values = [list(each) if each is not None else each for each in past_key_values]
        for key_value_layer in past_key_values:
            if key_value_layer is not None:
                key_value_layer[0] = key_value_layer[0][beam_idx]
                key_value_layer[1] = key_value_layer[1][beam_idx]

        # update input ids
        input = torch.cat([input, beam_words.unsqueeze(1)], dim=-1)
        length += 1
        context = torch.cat(
            [context, torch.ones((context.size(0), 1), dtype=torch.int, device=context.device)],
            dim=-1,
        )
        position = torch.cat([position, position[:, -1:] + 1], dim=-1)
        segment = torch.cat(
            [segment, segment[:, -1:]], dim=-1
        )  # segment id always the same as the previous token
        span = torch.cat([span, span[:, -1:]], dim=-1)

    # select the best hypotheses
    results = []
    for i, hypotheses in enumerate(generated_hyps):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        results.append(best_hyp)

    return results
