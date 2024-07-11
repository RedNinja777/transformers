import torch
from typing import Optional, Tuple
from itertools import accumulate
import numpy as np
import torch.nn.functional as F
from collections import UserDict

from transformers.generation_beam_search import BeamSearchScorer

from ..modeling_utils import replace


@replace(BeamSearchScorer)
class BeamSearchScorerV2(BeamSearchScorer):
    r"""
    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which standard beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        device (:obj:`torch.device`):
            Defines the device type (*e.g.*, :obj:`"cpu"` or :obj:`"cuda"`) on which this instance of
            :obj:`BeamSearchScorer` will be allocated.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
        num_beam_groups (:obj:`int`):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    """
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        use_reorder_cache_v2: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            batch_size,
            num_beams,
            device,
            length_penalty,
            do_early_stopping,
            num_beam_hyps_to_keep,
            num_beam_groups,
        )
        self.use_reorder_cache_v2 = use_reorder_cache_v2
        self.beams_offset = (
            (torch.arange(0, batch_size) * num_beams // self.num_beam_groups)
            .unsqueeze(1)
            .to(device)
        )
        self.cand_size = 2 * num_beams // self.num_beam_groups
        self.cand_offsets = torch.arange(0, self.cand_size).to(device)

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:

        cur_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0] // self.group_size

        next_tokens_id = next_tokens
        next_beams_id = next_indices
        effective_beam_id = next_beams_id + self.beams_offset

        if eos_token_id is not None:
            eos_mask = next_tokens.eq(eos_token_id)
        else:
            eos_mask = torch.zeros_like(next_tokens).bool()
        eos_effective_idx = torch.masked_select(
            effective_beam_id[:, : self.group_size], mask=eos_mask[:, : self.group_size]
        )

        finished_batch_idxs = []
        if self.use_reorder_cache_v2 and eos_effective_idx.numel() > 0:
            eos_effective_scores = torch.masked_select(
                next_scores[:, : self.group_size], mask=eos_mask[:, : self.group_size]
            )
            input_clone = input_ids.index_select(0, eos_effective_idx)
            unfin_offset = np.array(list(accumulate(map(int, self._done))))[
                np.array(list(map(int, self._done))) == 0
            ]
            for i in range(eos_effective_idx.size(0)):
                eos_idx = eos_effective_idx[i]
                eos_score = eos_effective_scores[i]
                unfin_batch_idx = eos_idx // self.group_size
                batch_idx = unfin_batch_idx + unfin_offset[unfin_batch_idx]
                if not self._done[batch_idx]:
                    self._beam_hyps[batch_idx.item()].add(
                        input_clone[i], eos_score.item()
                    )
                is_done = bool(self._done[batch_idx])
                self._done[batch_idx] = self._done[batch_idx] or self._beam_hyps[
                    batch_idx
                ].is_done(next_scores[unfin_batch_idx].max().item(), cur_len)
                if is_done != bool(self._done[batch_idx]):
                    finished_batch_idxs.append(unfin_batch_idx)

        if not self.use_reorder_cache_v2:
            eos_effective_scores = torch.masked_select(
                next_scores[:, : self.group_size], mask=eos_mask[:, : self.group_size]
            )
            input_ids_cpu = input_ids.cpu()
            eos_effective_idx_cpu = eos_effective_idx.cpu()
            eos_effective_scores_cpu = eos_effective_scores.cpu()
            for i in range(0, eos_effective_idx_cpu.size()[-1]):
                batch_idx = eos_effective_idx_cpu[i] // self.group_size
                if not self._done[batch_idx]:
                    self._beam_hyps[batch_idx.item()].add(
                        input_ids_cpu[eos_effective_idx_cpu[i]].clone(),
                        eos_effective_scores_cpu[i],
                    )
                self._done[batch_idx] = self._done[batch_idx] or self._beam_hyps[
                    batch_idx
                ].is_done(next_scores[batch_idx].max().item(), cur_len)

        if self.use_reorder_cache_v2 and len(finished_batch_idxs) > 0:
            new_batch_size = batch_size - len(finished_batch_idxs)
            batch_mask = torch.ones(batch_size).to(next_tokens_id)
            batch_mask[torch.tensor(finished_batch_idxs)] = 0
            batch_idxs = batch_mask.nonzero(as_tuple=False).squeeze(-1)
            eos_mask = eos_mask[batch_idxs]
            next_beams_id = next_beams_id[batch_idxs]
            self.beams_offset.resize_(new_batch_size, 1)
            effective_beam_id = next_beams_id.add(self.beams_offset)
            next_scores = next_scores[batch_idxs]
            next_tokens = next_tokens[batch_idxs]
            next_tokens_id = next_tokens_id[batch_idxs]
            before_batch_size = batch_size
            batch_size = new_batch_size
        else:
            before_batch_size = batch_size
            batch_idxs = None

        active_mask = torch.add(
            eos_mask.type_as(self.cand_offsets) * self.cand_size,
            self.cand_offsets[: eos_mask.size(1)],
        )
        _, active_hypos = torch.topk(
            active_mask, k=self.group_size, dim=1, largest=False
        )
        active_effective_beam_id = torch.gather(
            effective_beam_id, dim=1, index=active_hypos
        )
        active_scores = torch.gather(next_scores, dim=1, index=active_hypos)
        active_tokens = torch.gather(next_tokens_id, dim=1, index=active_hypos)
        beam_idxs = active_effective_beam_id.view(-1)
        beam_scores = active_scores.view(-1)
        beam_tokens = active_tokens.view(-1)

        if batch_idxs is not None:
            new_beam_idxs = (
                torch.arange(before_batch_size * self.group_size)
                .reshape(before_batch_size, self.group_size)
                .to(input_ids)
            )
            beam_idxs = new_beam_idxs[batch_idxs].reshape(-1)[beam_idxs]

        userdict = UserDict(
            {
                "next_beam_scores": beam_scores.view(-1),
                "next_beam_tokens": beam_tokens.view(-1),
                "next_beam_indices": beam_idxs.view(-1),
            }
        )
        if self.use_reorder_cache_v2:
            userdict["next_batch_indices"] = batch_idxs

        return userdict

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        unfin_offset = np.array(list(accumulate(map(int, self._done))))[
            np.array(list(map(int, self._done))) == 0
        ]
        if self.use_reorder_cache_v2:
            batch_size = len(unfin_offset)
        for batch_idx in range(batch_size):
            if not self.use_reorder_cache_v2 and self._done[batch_idx]:
                continue
            if self.use_reorder_cache_v2:
                final_batch_idx = batch_idx + unfin_offset[batch_idx]
            else:
                final_batch_idx = batch_idx
            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                effective_beam_id = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                self._beam_hyps[final_batch_idx].add(final_tokens, final_score)

        batch_size = len(self._beam_hyps)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            device=self.device,
            dtype=torch.float32,
        )
        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded: torch.LongTensor = input_ids.new(
            batch_size * self.num_beam_hyps_to_keep, sent_max_len
        )
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )
