# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""

import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json",
        "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/vocab.json",
        "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/vocab.json",
        "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/vocab.json",
        "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt",
        "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/merges.txt",
        "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/merges.txt",
        "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/merges.txt",
        "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-community/gpt2": 1024,
    "openai-community/gpt2-medium": 1024,
    "openai-community/gpt2-large": 1024,
    "openai-community/gpt2-xl": 1024,
    "distilbert/distilgpt2": 1024,
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    # what is bs? byte stream? in a byte number can be at most 255?
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    # ord("!") = 33, space is 32. ord("~") = 126. ord("¡") = 161 = \u00a1, ord("¬") = 172 = \u00ac. ord("®") = 174 = \u00ae, ord("ÿ") = 255 = \u00ff.
    # so all 0-9, A-Z, a-z, and common symbols are preserved their original ASCII symbols. All these are PRINTABLE. There are 188 of them. 
    cs = bs[:]
    n = 0
    for b in range(2**8):
        # 0 to 255.
        if b not in bs:
            # [0, 32], [127, 160], 173.
            # these are UNPRINTABLE characters between [0, 255], and are converted to PRINTABLE Unicode characters. There are 68 of them. 
            # n goes from 0 to 67, and so Unicode code point goes from 256 (\u0100) to 323 (\u0143)
            bs.append(b)
            # use Unicode code points [256 (\u0100), 323 (\u0143)]] to represent other unused chars in [0, 255). 0 is NULL and become 256 = \u0100 = Ā. 
            cs.append(2**8 + n)
            n += 1
    # ? convert Unicode code point to Unicode symbol. space (32) becomes 256 + 32 = 288 as its code point (\u0120), Ġ. '\t' becomes 256 + 9 = 265, \u0109 ĉ.
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
    # now bs and cs have 256 elements. bs are still numbers in 0-255. 
    # Unicode code point in cs are in ranges [33, 126], [161 (\u00a1), 172 (\u00ac)], [174 (\u00ae), 255 (\u00ff)], [256 (\u0100), 323 (\u0143)]. 


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
       Is word a str? No, It's more general, it's a tuple of (intermediate) tokens that are in the vocab. 
       Does this return bigrams of word? Yes, but a char of a word can be compound chars that were merged and placed in vocab earlier. 

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# does not have build_inputs_with_special_tokens() method. No bos or eos tokens added?
class GPT2Tokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding and tokenize methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
      Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
      the absence of a space at the beginning of a string:

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import GPT2Tokenizer

    >>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
            word just as any other word.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.add_bos_token = add_bos_token
        # the default behavior of GPT2 is to Not add bos or eos tokens.
        # GPT2 is mainly used to generate text so it would not make a lot of sense to add a EOS of a input prompt.
        # If one wants to finetune GPT2 for classification, he could manually add gpt2_tokenizer.eos_token to the input.

        # self.max_len_single_sentence = (self.max_len)  
		# no default special tokens - you can update this value if you add special tokens
        # self.max_len_sentences_pair = (self.max_len)  
		# no default special tokens - you can update this value if you add special tokens

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            # load json file into a dict, from token to id. Key is token.
            self.encoder = json.load(vocab_handle)
            ## every individual Unicode character in every key of the vocab must be a byte only and the Unicode character must be in the keys of self.byte_encoder.
            ## every tokens in the merge file must be one of the vocab.
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        # ? maps a byte integer to a Unicode character.
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            # in merges_file, first line is comment, last line is empty. Other lines are pairs ordered by frequency of appearing together.
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # ? {(,): idx}, the lower idx, the higher merge frequency.
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # what does this do? Define patterns to be matched.
        # RE pattern that matches some special tokens in English, space followed by letters or numbers, leading any whilespace followed by letters or numbers, 
        #                         any whitespaces followed by non-whitespace, any whitespaces.

        # ? matches 0 or 1 of preceding expression
        # + matches 1 or more preceding expression
        # * matches 0 or more preceding expression
        # \s matches whitespace (space, multiple spaces, tab, newline)
        # \p{L} matches a single Unicode code point in the category "letter".
        # \p{N} matches any kind of numeric character in any script.



        # This is kind of like a whitespace tokenizer but preserves the whitespace, which means we still assign white space special meaning to tokenize a string.

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        # Return the full vocabulary, including loaded vocab and added new tokens
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        """
        Called only in self._tokenize()  
        Given an input token represented in Unicode characters in self.byte_encoder,
        merge possible pairs of consecutive tokens until no more merges can be done,
        combine all final merged tokens with space and output.
        """
        if token in self.cache:
            return self.cache[token]
        # similar to list(), converts iterable to a tuple. 
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # lookup the most frequent possible merge
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    # merge the pair of into a token
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # starting from individual chars, keep merging until no more merges can be done based on the merge file. 
        # Then all final tokens are joined together into a string by a single space.
        word = " ".join(word)
        self.cache[token] = word
        return word

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text):
        """Tokenize a string."""
        # Does NOT do some cleanings, such as replace other white spaces, such as \t, \r, \n, multiple whitespaces with a single space?
        # Args:
        #   - add_prefix_space (boolean, default False):
        #      Begin the sentence with at least one space to get invariance to word order in GPT-2 (and RoBERTa) tokenizers. ??
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # token is RE matched part of the input text. Different from split, whitespaces are preserved in matched tokens. 
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
                # token is a Unicode char? No, a Unicode string. str.encode() Return an encoded version of the string as a bytes object. 
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens
        # list of tokens representing input text, after all merges are done. whitespaces are preserved. All tokens must be in vocab, self.encoder.

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # Converts a sequence of tokens (strings) in vocab into a single string of regular Unicode string .
        # all whitespaces are already preserved in tokens.
        text = "".join(tokens)
        # bytearray is a mutable sequence of bytes.
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        # This only saves vocabulary. Does not save added tokens, which is done is save_pretrained(). 
        # This should not be called by itself! It's already called inside save_pretrained(). 
        # Always use save_pretrained() instead!!

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)


__all__ = ["GPT2Tokenizer"]
