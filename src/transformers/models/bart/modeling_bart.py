# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BART model."""

import copy
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_bart import BartConfig

# @dataclass
# class Seq2SeqLMOutput(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None  # Languaged modeling loss. shape (1,), returned when `labels` is provided
#     logits: torch.FloatTensor = None  # Prediction scores of LM head (scores for each vocabulary token before SoftMax), shape (batch_size, sequence_length, config.vocab_size)
#     past_key_values: Optional[List[torch.FloatTensor]] = None  
#     # Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used to speed up sequential decoding. 
#     # List of length config.n_layers,  with each tensor of shape (2, batch_size, num_heads, sequence_length, embed_size_per_head). returned when se_cache=True is passed or when config.use_cache=True
#     decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Hidden-states of the decoder at the output of each layer plus the initial embedding outputs. (batch_size, sequence_length, hidden_size)
#     decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Attentions weights of the decoder (after softmax), tuple of shape (batch_size, num_heads, sequence_length, sequence_length)
#     encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # shape (batch_size, encoder_sequence_length, hidden_size)
#     encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Hidden-states of the encoder at the output of each layer plus the initial embedding outputs. tuple of shape (batch_size, encoder_sequence_length, hidden_size)
#     encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Attentions weights of the encoder (after softmax), tuple of shape (batch_size, num_heads, encoder_sequence_length, encoder_sequence_length)

if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-base"
_CONFIG_FOR_DOC = "BartConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "valhalla/bart-large-sst2"
_SEQ_CLASS_EXPECTED_LOSS = 0.0
_SEQ_CLASS_EXPECTED_OUTPUT = "'POSITIVE'"

# QuestionAsnwering docstring
_CHECKPOINT_FOR_QA = "valhalla/bart-large-finetuned-squadv1"
_QA_EXPECTED_LOSS = 0.59
_QA_EXPECTED_OUTPUT = "' nice puppet'"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
    # see all BART models at https://huggingface.co/models?filter=bart
]

 
BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
    # see all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # Does clone() share underlying data? No, separate data.
    # Unlike copy_(), clone() is recorded in the computation graph. Gradients propagating to the cloned tensor will propagate to the original tensor.
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    # why? -100 is only useful in labels in that pytorch automatically ignores them in loss. As input, should use padding token for tokens to ignore.
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

	# The following code is removed in new version; so no longer wrap the last non pad token (usually <eos>)
    # padding tokens are on the right; 
    # index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)  # sum() reduce dimension; result is [[example 1 actual_seq_len - 1], [example 2 actual_seq_len - 1], ...]
    # decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()  # index_of_eos has size (batch_size, 1). assign the eos_token_id to the first token position
    # prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()   # shifts to the right by 1
    # prev_output_tokens[:, 0] = decoder_start_tokens
    # return prev_output_tokens
    
    # what's the first token of input input_ids? bos? Yes if input_ids is built by tokenizer with 'add_special_tokens=True' option.
    # Done in tokenizer.prepare_for_model(), implemented in RobertaTokenizer.build_inputs_with_special_tokens(). 
    # In RobertaTokenizer, cls + token_ids_0 + sep [+ sep + token_ids_1 + sep]. 
    # In RobertaTokenizer, bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>"; "<s>": 0, "<pad>": 1, "</s>": 2. cls and bos are the same <s>; sep and eos are the same </s>.
    # In BartConfig, pad_token_id=1, bos_token_id=0,  eos_token_id=2. decoder_start_token_id = 2, 
    # So in Bart tokens, bos + token_ids_0 + eos [+ eos + token_ids_1 + eos]

    # why put eos as the first token in decoder input, and followed by a bos?
    # bos is added to the front of input_ids as usual. I imagin can use <bos> as decoder_start_token. 

    # torch.gather(input, dim, index, out=None, sparse_grad=False) -> Tensor  Gathers values along an axis specified by dim.
    # index must have the same size as input, other than the specified dim.
    # out tensor will have the same size as index tensor.

    # For a 3-D tensor the output is specified by:

    # out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0. means for every j and k, take value at index[i][j][k] on dim 0
    # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    # out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    # Example:

    # >>> t = torch.tensor([[1,2],[3,4]])
    # >>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
    # tensor([[ 1,  1],
    #         [ 4,  3]])

# These functions are removed in new version
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # input_ids is input to decoder (because only decoder has causal mask), aka query
    # past_key_values is the subsequence comes in front of input_ids, are fully visible to input_ids.
    # In training, this function is called only once and the complete causal mask is made for the complete target/decoder sequence.
    # In generation, this function is called in each token generation step. Depending on use_cache, past_key_values can be None or decoder subsequence before input_ids,
    #                but past_key_values + input_ids are the complete decoder sequence at this step.
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    # for example, tensor([0, 1, 2, 3, 4]) < tensor([[1], [2], [3], [4], [5]]), broadcast to 
    # tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]) < tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]])
    
    # can also possibly use torch.triu upper triangle matrix, starting from diagonal index

    mask = mask.to(dtype)
    # diagonal entries (itself) becomes 0, unmasked.

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    # causal mask shape [batch_size, 1, input_ids_length, input_ids_length + past_key_values_length]
    # The dim 1 is for num_heads.


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # This is padding token mask (not causal mask) from query (tgt) to key (src).
    # tgt and src can be for the same sequence (either encoder or decoder)
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # normal mask: 1 or True for real token, 0 or False for padding token
    # inverted_mask is the opposite: 0 or False for real token, 1 or True for padding token.
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
    # -inf for padding tokens, 0 for real tokens

    # A torch.finfo is an object that represents the numerical properties of a floating point torch.dtype, (i.e. torch.float32, torch.float64, and torch.float16). 


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        # why this hack?
        self.offset = 2
        # why 2? because in Bart, pad_token_id = 1, so 2 = pad_token_id + 1
        super().__init__(num_embeddings + self.offset, embedding_dim)
        # nn.Embedding is to encode each index in [0, ..., num_embeddings - 1] as an embedding vector of length embedding_dim.
        # With padding_idx set, the embedding vector at padding_idx is initialized to all zeros. 
        # However, note that this embedding vector of padding_idx can be modified afterwards, e.g., using a customized initialization method, and thus changing the vector used to pad the output. 
        # The gradient for this vector from Embedding is always zero, though.

        # Embedding.weight (Tensor) � the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from N(0,1) 

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        # the position is our current step in the decoded sequence
        # if use_cache (i.e., past_key_values_length > 0), input_ids will be just a single token of last position
        # use_cache is True only in generation, right?

        # if not use_cache, starts at 0, ends at seq_len-1
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)
        # starting position id is self.offset (which is set to self.pad_token_id + 1 in BartConfig) instead of 0, 
        # so no position id will be the same as self.pad_token_id, other than padding tokens's positon.
        # why is this hack important?
        # so when position idx for padding tokens is set to self.padding_idx, which is done is create_position_ids_from_input_ids,
        # these padding tokens' position embedding is not updated either.


class BartScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        # encoder has only self attention; decoder has extra cross attention
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BartConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.out_proj is like a simple version of BertSelfOutput layer (without dropout, layernorm, residual). 
        # Only does a linear transform after doing attention. 
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # input tensor has regular shape (batch_size, seq_len, hidden_size)
        # this function reshape to (batch_size, num_heads, seq_len, head_size)
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # tensor.contiguous() Returns a contiguous in memory tensor containing the same data as self tensor. 
        # If self tensor is already in the specified memory format, this function returns the self tensor (no op). Otherwise return a copy.


    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # hidden_states is input sequence to this BartAttention module, shape  (batch_size, tgt_len, d_model). tgt is query when using attention.
        # !! key_value_states is encoder_outputs. It's only not None if this is a cross attention in a decoder
        # past_key_value is alwyas None if this is self attention in encoder
        # If this is self attention in decoder, past_key_value and hidden_states combined is the complete sequence.
        # if this is cross attention in decoder, past_key_value is None during training or very first step of generation. 
        #   During the first step of generation, past_key_value saves the reshaped projected encoder_outputs for reuse in subsequent steps of generation, if use_cache.
        # !! attention_mask, if not None, can be query/target sequence causal mask + padding mask. shape is broadcastable to (batch_size, num_heads, tgt_len, src_len). 
        # attention_mask should be prepared outside of this BartAttention module.
        # Note padding mask is always for key, not for query. So even if query is [MASK] token, it still calcualtes attention with key tensor. 

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # tgt is the query
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # shape [batch_size, tgt_len, hidden_size], and always scaled by square root of head size

        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # this is cross attention (and of course in decoder), but past_key_value != None. This happens during subsequent steps of generation.
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # this is cross attention (and of course in decoder), but past_key_value=None. This happens during training or the first step of generation.
            # this same module object will be called in subsequent steps in generation, so save reshaped projected encoder_outputs in past_key_value in this first step.
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # this is self attention in a decoder, because past_key_value is always None in encoder self attention
            # concate past_key_value and current input hidden_states (which is query) to form complete key of self attention
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            # this is self attention (can either be in encoder, or the first step of decoder), because past_key_value is None
            # if this is self attention in decoder, past_key_value will save reshaped projected tgt sequence for subsequent steps if in generation
            # so self_attention query/key/value sequence are based on the same input sequence hidden_states
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        # query_states/key_states/value_states shape (batch_size * num_heads, seq_len, head_size)
        src_len = key_states.size(1)
        # query_states is projected from query which is tgt, the input hidden_states to this attention module.
        # key_states is projected from key, which can be past_key + query if self attention, or encoder last layer output if cross attention
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        # attn_weights shape (batch_size * num_heads, q_len, k_len)
        # torch.bmm() Performs a batch matrix-matrix product of two 3-D tensors each containing the same number of matrices.
        # torch.bmm() does not broadcast. For broadcasting matrix products, see torch.matmul().

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )


        # # during beam search, it potentially saves memory used by encoder_outputs if we don't duplicated every src_seq encoder_outputs by beam_size times
        # if self.encoder_decoder_attention == True and bsz != kv_bsz:
        #     # assuming query bsz is a multiple (num_beams) of kv_bsz
        #     attn_weights = torch.einsum('bxhtd,bhsd->bxhts', q.view(kv_bsz, -1, self.num_heads, *q.size()[1:]), k.view(kv_bsz, self.num_heads, *k.size()[1:]))
        #     attn_weights = attn_weights.reshape(-1, *attn_weights.size()[-2:])
        # else:
        #     attn_weights = torch.bmm(q, k.transpose(1, 2))


        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            # attn_mask is query/target attends to key/src. It can have key padding mask and query to key causal mask combined.
            # shape is broadcastable to (bsz, num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            # why? 
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # attn_probs shape (bsz * num_heads, q_len, k_len)
        # is training=self.training needed? Yes, only apply dropout if training=True and scale output in training.

        # # during beam search, if encoder and decoder have different batch size dim
        # attn_probs = attn_probs.to(v.dtype)  # einsum() does not promote data types when summing.?
        # if self.encoder_decoder_attention == True and bsz != kv_bsz:
        #     attn_output = torch.einsum('bxhts,bhsd->bxhtd', attn_probs.view(kv_bsz, -1, self.num_heads, *attn_probs.size()[1:]), v.view(kv_bsz, self.num_heads, *v.size()[1:]))
        #     attn_output = attn_output.reshape(-1, *attn_output.size()[-2:])
        # else:
        #     attn_output = torch.bmm(attn_probs, v)

        # To understand Einstein summation convention einsum(), remember these three rules: https://ajcr.net/Basic-guide-to-einsum/
        #   1. Repeating letters between input arrays means that values along those axes will be multiplied together. The products make up the values for the output array.
        #   2. Omitting a letter from the output means that values along that axis will be summed.
        #   3. We can return the unsummed axes in any order we like.

        # value_states shape (bsz * num_heads, k_len, head_dim)
        attn_output = torch.bmm(attn_probs, value_states)
        # attn_output shape (bsz * num_heads, q_len, head_dim)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        # attn_output is reshaped to (bsz, tgt_len, hidden_dim)

        # hidden vectors after attention go through a Linear layer, by default Linear layer has bias.
        # It's done without dimension change; just to let attention focus on attention, and use extra layer to prepare for another layer to compute logits
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
        # attn_output is the hidden vector representation of input hidden_states after this Attention operation, shape (bsz, tgt_len, embed_dim)
        # attn_weights_reshaped is attention probablity before any dropout, (bsz, num_heads, tgt_len, src_len)
        # past_key_value will be passed to the next call of this same BartAttention instance, which may happen in decoder in generation,
        #   past_key_value may be encoder_outputs if this is cross attention, or decoder sequence up to current input


class BartFlashAttention2(BartAttention):
    """
    Bart flash attention module. This module inherits from `BartAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # BartFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError("BartFlashAttention2 attention does not support output_attentions")

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        # get query proj
        query_states = self._reshape(self.q_proj(hidden_states), -1, bsz)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0].transpose(1, 2)
            value_states = past_key_value[1].transpose(1, 2)
        elif is_cross_attention:
            # cross_attentions
            key_states = self._reshape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._reshape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
            value_states = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
        else:
            # self_attention
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BartSdpaAttention(BartAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions or layer_head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "BartModel is using BartSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention"
                ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, tgt_len, bsz)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        is_causal = True if self.is_causal and attention_mask is None and tgt_len > 1 else False

        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
            is_causal=is_causal,
        )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


BART_ATTENTION_CLASSES = {
    "eager": BartAttention,
    "sdpa": BartSdpaAttention,
    "flash_attention_2": BartFlashAttention2,
}


class BartEncoderLayer(nn.Module):
    # EcoderLayer is applied after Embedding layer in BartModel.
    # Encoder only has self attention, no cross attention
    # The only place that does nonlinear activation in one EncoderLayer is after first dense layer (which is right after Attention layer) that increases to intermediate hidden size.
    def __init__(self, config: BartConfig):
        super().__init__()
        # BART's naming for hidden_size
        self.embed_dim = config.d_model

        self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            # by default, is_decoder=False, bias=True
        )
        # LayerNorm normalizes over the last several dimensions which have to be of the shape specified by argument normalized_shape
        # If a single integer is used, it is treated as a singleton list, and this module will normalize over the last dimension which is expected to be of that specific size.
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # config.dropout is other non-attention/non-activation layer's dropout probability, 
        # Inside attention layer it has its own dropout probability of attention weights specified by config.attention_dropout
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        # activation_dropout is used after intermediate layer to increase hidden size and nonlinear activation
        self.activation_dropout = config.activation_dropout
        # config.encoder_ffn_dim is intermediate hidden size, usually 4 times of hidden size
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.

        Returns:
            encoded output of shape `(batch, seq_len, embed_dim)` and possibly attn_weights
        """
        # hidden_states is input to BartEncoderLayer (and the first EncoderLayer comes after Embedding), shape [batch_size, seq_len, embed_dim]
        # In encoder, attention_mask is purely padding mask, 1 for padding tokens, 0 for actual tokens. shape []
        # Note padding mask is always for key, not for query or value. So even if query is [MASK] token, it still calcualtes attention with key tensor. 
        
        # keep original input for residual connection
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # hidden_states is attention layer output, shape (batch_zie, query_len, embed_dim)
        # encoder always has past_key_value (the third returned item) at None

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # residual connection after dropout and before layer norm. Because adding residual might change mean/var, so apply layer norm after residual
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # Is it common to apply dropout between two FC layers?
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # no layer norm between two FC layers. Is it good to have one?
        # same as BERT/Transformer, dropout are done after the 2nd Linear FC layer.
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # residual connection after dropout and before layer norm. Because adding residual might change mean/var, so apply layer norm after residual
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # A torch.finfo is an object that represents the numerical properties of a floating point torch.dtype, (i.e. torch.float32, torch.float64, and torch.float16). Similar to numpy.finfo.
            # A torch.iinfo is an object that represents the numerical properties of a integer torch.dtype (i.e. torch.uint8, torch.int8, torch.int16, torch.int32, and torch.int64). similar to numpy.iinfo.
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            #  torch.clamp(input, min, max, out=None) Clamp all elements in input into the range [ min, max ] and return a resulting tensor

        # hidden_states shape (batch_zie, query_len, embed_dim)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(nn.Module):
    # decoder has both self attention and cross attention
    # 
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # decoder self attention
        self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
            # by default bias=True
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # encoder decoder cross attention
        self.encoder_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            # by default bias=True
        )
        # layer norm after cross attention
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # config.decoder_ffn_dim is intermediate hidden size. increase dim 4X and nonlinear activation before decrease back
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # hidden_states is input embeddings to this decoder, (batch_size, tgt_len, emb_dim)
        # attention_mask is for hidden_states which is input to this decoder. It's combined padding mask and causal mask. It should be prepared outside of BartDecoderLayer.
        # encoder_hidden_states is the last encoder layer output
        # encoder_attention_mask is purely padding mask for encoder sequence
        # past_key_value input to decoder is, if not None, a tuple of 4 that has both previous decoder sequence, and encoder outputs.
        # If in training, past_key_value is None. And there is only one call to this decoder instance, 
        # If in generation, the first step receives past_key_value is None, update it, and reuse it subsequent steps. 
        # is past_key_value updated during this decoder forward()? Yes, updated in SelfAttention module. Reused if this decoder instance is called again, in generation.
        # EncoderLayer module does not have past_key_value.

        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # present_key_value is updated in decoder BartAttention and returned if use_cache=True. It might get reused by subsequent calls of this decoder instance, in generation.

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # residual connection after dropout and before layer norm. Because adding residual might change mean/var, so apply layer norm after residual
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # There is no nonlinear activation between self attention and cross attention

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            # !! the '+' operator on tuple/list is concatenation, not summation.
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # no layer norm between two FC layers. Is it good to have one?
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # residual connection after dropout and before layer norm. Because adding residual might change mean/var, so apply layer norm after residual
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # this is where (1 out of 3 places) use_cache is used.
        # 
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    # Head is applied on top of BartModel output.
    # BartModel output is always the output of BartDecoder.

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states is output of last  layer, possibly only of a single token.
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        # typically use tanh nonlinear activation just before the last Linear layer that output num_classes logits
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartPreTrainedModel(PreTrainedModel):
    # are class attributes inherited? Yes. For example, config_class = BartConfig is inherited in all descendants of PretrainedBartModel.
    config_class = BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"BartEncoderLayer", r"BartDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        # Isn't similar init done in trainer? With the exception of keeping embedding module padding_idx embedding at 0.
        # standard deviation of normal distribution that's used to initialize parameter weights
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            ## !! nn.Embedding module has no bias, only weight !!
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # where is padding_idx attribute set? In both BartEncoder and BartDecoder, self.padding_idx = embed_tokens.padding_idx
                # random initialization have changed embedding vector of padding_idx. Reset it to zero vector.
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        # why need this?
        
        # pad_token_id = 1 in Bart. decoder_start_token_id = eos_token_id = 2.
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # bos_token_id=0, eos_token_id=2, 
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedBartModel(BartPreTrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


class BartPretrainedModel(BartPreTrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BartConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, BartForConditionalGeneration

    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    >>> ARTICLE_TO_SUMMARIZE = (
    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    ... )
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
    ```

    Mask filling example:

    ```python
    >>> from transformers import AutoTokenizer, BartForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    >>> TXT = "My friends are <mask> but they eat too many carbs."
    >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    >>> logits = model(input_ids).logits

    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    >>> probs = logits[0, masked_index].softmax(dim=0)
    >>> values, predictions = probs.topk(5)

    >>> tokenizer.decode(predictions).split()
    ['not', 'good', 'healthy', 'great', 'very']
    ```
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# These functions are removed in new version
def _reorder_buffer(attn_cache: Dict, new_order) -> Dict:
    # only used in beam search.
    # attn_cache ? is a dict of all layers?
    # new_order is 
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache
    # torch.index_select(input, dim, index, out=None) -> Tensor
    # Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a 1-D LongTensor.
    # The returned tensor has the same number of dimensions as the original tensor (input). 
    # The dimth dimension has the same size as the length of index; other dimensions have the same size as in the original tensor.

    # Differenece from torch.gather? gather is a multi-index selection method, while index_select 

    # NOTE: The returned tensor does not use the same storage as the original tensor. 
    # If out has a different shape than expected, we silently change it to the correct shape, reallocating the underlying storage if necessary. 

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


# Public API
def _get_shape(t):
    return getattr(t, "shape", None)

# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))



class BartEncoder(BartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # BartEncoder is different from BertEncoder in that this includes Embedding layer. 
    # So input to BartEncoder can be either a sequence of input_ids, or input_embeddings.

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # embed_tokens is an instance of nn.Embedding module; it's NOT a tensor
        # it's used to turn input_ids to embeddings.
        # position embeddings are added inside BartEncoder
        super().__init__(config)

        self.dropout = config.dropout
        # probability that an entire EncoderLayer is dropped; defaults to 0.0 which means no layer drop
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        # ModuleList can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all Module methods.
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # input_ids is a tensor of token ids, shape [batch_size, len of input_ids]
        # attention_mask is encoder input_ids padding mask; shape (batch_size, len of input_ids). Usually 1 for actual tokens, and 0 for padding token
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # input_ids shape (batch_size, seq_len)
        # first get input embeddings and position embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        # Sum token embeddings and position embeddings; 
        # like RoBERTa, BART does not use segment/token_type embedding
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # .training is an attribute of nn.Module. By default it's set to True.
        # Where is self.training set?

        # expand attention_mask
        # the input attention_mask is only padding mask
        if attention_mask is not None:
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self._use_sdpa and head_mask is None and not output_attentions:
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # concatenation
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # layer drop is only used in training
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    # gradient_checkpointing is only used in training
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        
        # add the last encoder layer's output to encoder_states list
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            # hidden_states is the last encoder layer's output hidden vector
            # encoder_states is all encoder layers' output hidden vectors, including embedding layer's output as first element, and x as last element.
            # all_attentions are all encoder layers' self attention weights
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # Like BartEncoder, BartDecoder also includes Embedding layer. 
    # So input to BartEncoder can be either a sequence of input_ids, or input_embeddings.

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # embed_tokens is an instance of nn.Embedding that turns input_ids into input embeddings
        super().__init__(config)
        self.dropout = config.dropout
        # probability of dropping a DecoderLayer completely; default to 0 in BartConfig which means no drop
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # return instance not tenor?
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # set instance not tenor?
        self.embed_tokens = value

    # This method is removed in new version
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # This returns decoder self attention mask (include padding and causal masks), not cross attention mask.
        # input attention_mask is padding mask of complete decoder input sequence, which include past_key and current input
        # input_shape is shape of input_ids, the input to this BartDecoder.
        # In generation:
        #  - if to generate the first token, input_ids is either <bos> or decoder prompt string, and past_key_values_length = 0
        #  - if to generate subsequent tokens and use_cache, input_ids is a single token, the last generated one. past_key_values

        # !! past_key_values_length does include decoder start token <bos> or decoder prompt string.

        # combined_attention_mask has both causal mask and padding mask
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # only need causal mask if input_ids is longer than 1, otherwise, it can trivially attend to all tokens in front of it which is None
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

            # !! decoder prompt also use causal mask to calculate their embeddings. Earlier prompt tokens can't attend to later prompt tokens!

        # input attention_mask is padding mask for the entire decoder input sequence, which include past_key and current input
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        # !! if attention_mask is not provided, assume there is no padding mask !!

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Includes several features from "Jointly Learning to Align and Translate with Transformer Models" (Garg et al.,
        EMNLP 2019).

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # in BartDecoder, input_ids is for decoder, shape [batch, decoder input_ids length]
        # !! input attention_mask is simply padding mask of decoder inputs !! causal mask is prepared inside this forward() by calling self._prepare_decoder_attention_mask()
        # encoder_hidden_states is the last encoder layer output hidden states
        # encoder_attention_mask is padding mask of encoder input ids and should be provided to model
        # past_key_values is a dict of all layers of DecoderLayer, Each layer has past_key_value
        # during training, for each training example, BartModel, and so BartEncoder and BartDecoder, is called only once. So past_key_values is not updated.
        # use_cache is only used in decoder, not encoder. In decoder it's only useful in generation scenario where the same decoder instance is called multiple times.

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Bart config.use_cache is default to True
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # past_key_values is for all layers; past_key_values[0] is past_key_value of first layer; 
        # past_key_values[0][0] is past_key states, shape [batch_size, num_heads, key_len, head_size]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input)

        # !! only need to input padding mask of decoder input ids; causal mask will be prepared by this
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions and cross_attn_head_mask is None:
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self._use_sdpa and cross_attn_head_mask is None and not output_attentions:
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        # sum token embeddings and position embeddings
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                # concatenation. embeddings are included
                all_hidden_states += (hidden_states,)
            # layer drop is only used in training
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # gradient_checkpointing is only useful in training when running backward to calculate gradients.
                # gradient_checkpointing saves intermediate activations to disk and then discards to save memory. These activations are recalculated during backward.

                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
                # !! this is to set past_key_value=None? gradient_checkpointing will discard all in-memory activations
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            # this is where (2 out of 3 places) use_cache is used.
            if use_cache:
                # use_cache is True only in generation. correct?
                # Not correct. Bart config.use_cache is default to True. But cache is only useful in generation when this same decoder instance is called multiple times
	            # At the first step of generation, input_ids is just decoder_input_ids, which is either <bos> or decoder prompt string.
	            # In subsequent steps, only use the last token of the input_ids sequence. 

                # concate all layers past_key_value for next potential call of this decoder
                # take the "past_key_value" in layer_outputs
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # this is where (3 out of 3 places) use_cache is used.
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
            # hidden_states is the last decoder layer's output hidden vector
            # next_cache is (encoder_key_value_states, next_decoder_cache) 
            # all_hidden_states is all decoder layers' output hidden vectors, including embedding layer's output as first element, and x as last element.
            # all_self_attns are all decoder layers' self attention weights

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class BartModel(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # config.vocab_size does not include possible manually added new tokens? Then embeddings does NOT have correct size?
        # It's intialized this way, but will be resized by resize_token_embeddings() in PretrainedModel
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = BartScaledWordEmbedding(vocab_size, config.d_model, padding_idx, embed_scale=embed_scale)
        # nn.Embedding module has no bias parameters!
        # nn.Embedding.weight has shape (num_embeddings, embedding_dim)!
        # padding_idx defaults to None; If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the padding_idx.
        # With argument padding_idx given a value, the embedding vector at padding_idx is initialized to all zeros. 
        # This vector can be modified afterwards if for example using a customerized initializer
        # However, the gradient for this embedding vector (for padding_idx) from Embedding is always zero. So it's not trained. 

        # pass an Embedding module object to both Encoder and Decoder, 
        # so Encoder and Decoder input can be input_ids
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        # encoder and decoder use the same Embedding object, self.shared, 

        # Initialize weights and apply final processing
        self.post_init()
        # Do NOT forget to do this in all subclasses!! Readlly needed? Doesn't subclass's __init__() call super().__init__() first? 
        # That's true, but remember super().__init__() is the first statement called, and only affect super()'s modules. Has no impact on this class's new defined modules.
        # init_weights() first calls self.apply(self._init_weights), then calls self.tie_weights() to tie input and output word embedding weights.

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_input_embeddings(self):
        # return instance, not tensor?
        return self.shared

    def set_input_embeddings(self, value):
        # set value is instance, not tensor?
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # input_ids is for encoder
        # Can input_ids be None?
        # In classification scenario, the same sequence is input to both Bart encoder and decoder.
        # In seq-2-seq scenario,encoder input is source seq, decoder input is target seq (which is same as labels) shifted by one token.

        # who provides decoder_input_ids? 
        # For a seq2seq model, encoder and decoder have different inputs, Dataset should provide it in the batch, e.g., decoder_input_ids = shift_tokens_right(labels, pad_token_id)
        # For a NLU problems such as sequence/token classification, encoder and decoder have the same inputs (only difference is shift), 
        # then batch data only needs to provide input_ids, attention_mask, and optionally labels for training/evaluation,
        #  and then decoder_input_ids can be prepared by shift_tokens_right() on input_ids
        #  and then decoder_attention_mask can be prepared inside BartDecoder 

        # In generation (not in training or evaluation!!):
        # when this forward() is called to generate the first token, past_key_values is None, and decoder_input_ids is either <bos> or generation prompty string;
        # when generating subsequent tokens and use_cache, past_key_values is not None, decoder_input_ids is just a single token, the last generated one! 

        # when is decoder_input_ids None? User may not supply it (because it's NLU task and user assumes only encoder is needed). 
        # Then shift_tokens_right() called below will make decoder_input_ids by shifting input_ids.		

        # attention_mask is padding mask for input_ids which is for encoder
        # decoder_attention_mask is padding mask for decoder_input_ids which is for decoder

        # who provides decoder_attention_mask? If DataLoader does not provide it, _prepare_bart_decoder_inputs() will do

        # what if attention_mask is None? Then BART assume there is no padding tokens in input_ids, all tokens will be used in attention. 

        # what's in past_key_values?
        # past_key_values is Tuple[Tuple[torch.Tensor]] 
        # the outer tuple has length config.n_layers, each layer is also a tuple (the inner tuple);
        # the inner tuple has multiple (>= 2) tensors; the first 2 tensors are past key sequence and past value sequence,
        # and both key and value tensors have shape (batch_size, num_heads, past_sequence_length, embed_size_per_head).

        # past_key_values is only used by Decoder, and also output by Decoder. It's used by Decoder when doing token-by-token generation, to reduce redundant regeneration.


        # use_cache is only used in decoder.


        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        # what if input_ids is None?
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            # How about decoder_attention_mask? It should include both decoder input padding mask and causal mask
            # causal mask is prepared inside BartDecoder by _prepare_decoder_attention_mask().
            # But padding mask needs to be passed in as decoder_attention_mask?
            # Think about it. 
            # In training when labels must be provided (and -100 used for padding tokens), even if no decoder padding mask is provided, they are usually padding on the right, and not used in loss.
            # In generation, decoder padding tokens are added on the right, and does not matter.
            # But it's still better to provide decoder_attention_mask as padding mask for training. 


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # BartConfig, as well as, base class PretrainedConfig, use_cache default to True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if encoder_outputs is already provided, there is no need to run encoder again.
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # encoder_outputs is BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)
            # encoder_outputs can be either a tuple or OrderedDict, and [0] is its first element which is the last encoder layer's output 
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# The "Conditional" part in the name means this class is for encoder-decoder model, i.e., the typical Bart model. 
# There is also an uncommon use of BART, which uses Bart decoder only, like GPT2. This class is BartForCausalLM.
@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class BartForConditionalGeneration(BartPreTrainedModel, GenerationMixin):
    # No need here since it's already defined in base class PretrainedBartModel?
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # Used in PretrainedModel.from_pretrained(), some models may have keys that are not in the state by design, removing them before needlessly warning the user.
    # But why need to add these keys?

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # buffer is module's persistent state that is not a model parameter to be trained. 
        # register_buffer Adds a persistent state (buffer is a tensor) to the module.  
        # This is typically used to register a buffer that should not to be considered a model parameter. 
        # Buffers can be accessed as attributes using given names.
        # One reason to register a tensor as a buffer is to be able to serialize the model and restore all internal states, including parameters and buffers. 
        # Another reason is that all buffers and parameters will be pushed to the device, if called on the parent model. 

        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # nn.Embedding does not have bias. It's fine for input embeddings (why?), but may not be the best for output embeddings.
        # But since input and output embeddings are tied, their weight matrix are the same. So can manually add bias to output embeddings

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # add embedding vectors for new tokens with initialized values, or truncate un-needed tokens
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # replace registered buffer done in __init__
        self.register_buffer("final_logits_bias", new_bias)
        # bias is not trained as they are not parameters?? Not. Kept at all 0s. Just used as placeholder in F.linear().
        # correct?

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        # input_ids is for encoder
        # attention_mask is padding mask for input_ids. What if it's None? Then BartModel assumes there is no padding tokens in input_ids. So it should be provided!
        # decoder_input_ids is for decoder
        # decoder_attention_mask is padding mask for decoder_input_ids. decoder causal mask is prepared inside model, but not padding mask.
        # But think about it, it's not that serious is decoder_attention_mask is not provided.
        # In training, if padding is on the right, then padding mask has labels -100 and is ignored 
        # in generation, decoder_attention_mask does not matter.
        # But it's still a good idea to provide decoder_attention_mask.

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            # if labels is provided, assume this task is for training or validation, not generation.
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # if labels is None and decoder_input_ids is None, then decoder_input_ids remains None and passed to BartModel.forward().
        # Then inside BartModel.forward(), it creates decoder_input_ids by shifting input_ids to the right. This is like using Bart as an encoder only.
        # This is wrong behavior for generation. Luckily generate() will create a decoder_input_ids. 
        # !! It's best to provide decoder_input_ids when calling this forward

        # call base_model BartModel.forward
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # outputs is the returned value of base_model BartModel's forward(), which is BartDecoder's forward() output.
        # Seq2SeqModelOutput(
        #     last_hidden_state=decoder_outputs.last_hidden_state,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        # What's the point of adding bias? Is it fixed at 0?
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        # self.model.shared is nn.Embedding(vocab_size, config.d_model, padding_idx)
        # nn.Embedding.weight has shape (num_embeddings, embedding_dim)!
        # in F.linear(input, weight, bias=None), weight is (out_features,in_features), and return input * transposed weight + bias
        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            # equivalent to nn.functional.cross_entropy(input, target), combines log_softmax (nn.LogSoftmax()) and nll_loss (nn.NLLLoss()) in a single function. 
            # torch.nn.CrossEntropyLoss(weight: Optional[torch.Tensor] = None, ignore_index: int = -100, reduction: str = 'mean')

            # By default, ignore_index=-100
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

	# removed in new version
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # This is called in geneate().
        # the passed in decoder_input_ids is always the full decoder input sequence, including <bos>/prompt and any generated tokens.

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            # padding mask for input_ids/encoder_outputs
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        # this returned dict is all the inputs to this module's forward() function. Note there is no "labels" as this is only for generation!
        # does not prepare decoder_attention_mask, which will be prepared on the fly in BartModel.forward()

        # does not take the last token of decoder_input_ids only when use_cache?? 
        # not in this function. but it's done in BartDecoder forward(), if use_cache=True. Actual work is done in Attention class forward()
        # better to do it here like GPT2 and check if past is None to handle the first generated tokens before updating past?

        # decoder_input_ids sequence never shrinks, only grows? Yes, but only the last token is used inside BartDecoder.


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # inside shift_tokens_right(), -100 which is used as ignored tokens for labels is replaced by pad_token_id
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # removed in new version
    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        # only force generation of bos and eos tokens
        # cur_len is the current length of input_ids to decoder
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            # the current generated seq only includes decoder_start_token_id.
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    # removed in new version
    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")
        # Note that self.config.vocab_size is already resized to include new added tokens in resize_token_embeddings() method.
        # scores are logits, before softmax to probability


    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # only used in beam search. 
        # what for?
        # During beam search, at one step, there are num_beams input_ids (and also past_key_values) per example.
        # The top num_beams candidate next_token may be all from a single beam!
        # So after selecting these top next_token, input_ids will be all the same, the beam that generated these tokens.

        reordered_past = ()
        for layer_past in past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
        
        # beam_idx is actually batch_beam_idx of input_ids dim 0
        # return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)
        # past is of all layers 

        # torch.index_select(input, dim, index, out=None) Returns a NEW tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor, 0d or 1d.
        # The returned tensor has the same number of dimensions as input. The dim dimension has the same size as the length of index; other dimensions have the same size as in input.
        # index_select() is quite like index slicing using [_,:] operator
        # NOTE The returned tensor does NOT use the same storage as the original tensor. COPY data and waste? Have to copy data?

        # index_select() simply slices the original tensor in one dimension.

        # Difference with torch.gather(input, dim, index, out=None, sparse_grad=False)?
        # gather() can do more complicated indexing. If input is an n-dimensional tensor with size (x_0, x_1..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1}) and dim = i, 
        # then index must be an n-dimensional tensor with size (x_0, x_1, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1}) where y >= 1.
        # !! output tensor will have the same size as index.!!
        # For a 3-D tensor the output is specified by:
        # out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        # out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

        # torch.masked_select(input, mask, out=None) Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.
        # The shapes of the mask tensor and the input tensor don't need to match, but they must be broadcastable.
        # NOTE The returned tensor does NOT use the same storage as the original tensor.




@add_start_docstrings(
    """
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    BART_START_DOCSTRING,
)
class BartForSequenceClassification(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
        	# input_dim
            config.d_model,
            # inner_dim
            config.d_model,
            config.num_labels,
            # probability of dropout
            config.classifier_dropout,
        )
        # BartClassificationHead has two Linear layers. The first Linear layer has tanh activation

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state
        # shape (batch_size, seq_len, hidden_size)

        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)
        # input_ids is input to encoder? so eos_mask is bool tensor of encoder input?
        # but hidden_states is decoder output?

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
            # why requires this? because use embedding output of the last <eos> token for sequence classification
	    # take the last <eos> token if there are multiple
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        # !! the hidden vector that represents each example in the batch can be located in diferent position of the sequence. 
        # it's the position of the <eos> in the input_ids
        # Not like Bert, uses the first token of every example.
        # sentence_representation shape (batch_size, hidden_size)

        logits = self.classification_head(sentence_representation)
        # logits shape [batch_size, num_labels]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # loss is scalar tensor
        # logits shape [batch_size, logits]
        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings(
    """
    BART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BART_START_DOCSTRING,
)
class BartForQuestionAnswering(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=Seq2SeqQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=_QA_EXPECTED_LOSS,
        expected_output=_QA_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqQuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            # start_positions and end_positions are labels for qa. 
            # not use cache in training/validation
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last DECODER layer output
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        # split by last dim, each split has size 1 in the last dim
        start_logits, end_logits = logits.split(1, dim=-1)
        # shape (batch_size, seq_len)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            # ?? where is start_positions split?
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # loss is scalar tensor
        # start_logits and end_logits shape (batch_size, seq_len)
        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BartDecoderWrapper(BartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = BartDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


@add_start_docstrings(
    """
    BART decoder with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    BART_START_DOCSTRING,
)
class BartForCausalLM(BartPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = BartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BartForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        >>> model = BartForCausalLM.from_pretrained("facebook/bart-base", add_cross_attention=False)
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
        >>> list(logits.shape) == expected_shape
        True
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past



######################################
# simple example to pretrain BART
######################################
# https://github.com/huggingface/transformers/issues/4151

# tok = BartTokenizer.from_pretrained("facebook/bart-large")
# model = BartForConditionalGeneration(BartConfig())  # construct a (randomly initialized) BART model

# input_batch = ["My dog is <mask></s>", "It loves to play in the <mask></s>"]
# decoder_input_batch = ["<s>My dog is cute", "<s>It loves to play in the park"]
# labels_batch = ["My dog is cute</s>", "It loves to play in the park</s>"]

# input_ids = tok.batch_encode_plus(input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
# decoder_input_ids = tok.batch_encode_plus(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
# labels = tok.batch_encode_plus(labels_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids

# loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]



__all__ = [
    "BartForCausalLM",
    "BartForConditionalGeneration",
    "BartForQuestionAnswering",
    "BartForSequenceClassification",
    "BartModel",
    "BartPreTrainedModel",
    "BartPretrainedModel",
    "PretrainedBartModel",
]
