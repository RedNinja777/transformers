# coding=utf-8

import copy
import functools
import gc
import inspect
import os
import random
import re
import math
import json
import collections
# import matplotlib.pyplot as plt
import warnings
from contextlib import contextmanager
import dataclasses
from dataclasses import dataclass
import threading
import time
import tracemalloc
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import logging

import numpy as np
import torch



logger = logging.get_logger(__name__)



########################################################################################################
######### My Trainer utils begin #################

def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    # for example, in Bart config,
    #   "task_specific_params": {
    #     "summarization": {
    #       "early_stopping": true,
    #       "length_penalty": 2.0,  # length_penalty > 1 will penalize shorter seq, because score is logprob which is < 0.
    #       "max_length": 142,  # for src or tgt?
    #       "min_length": 56,
    #       "no_repeat_ngram_size": 3,
    #       "num_beams": 4
    #     }
    #   },
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"The task is {task}. Add/update these model.config parameters using task specific params in model.config object: {pars}")
        model.config.update(pars)


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)   # remove pegasus newline char
    return "\n".join(nltk.sent_tokenize(x))



def is_local_master(args):
    # args.local_rank is within [0, num_GPUs_on_this_node - 1], passed in by pytorch launch utility.
    return args.local_rank in [-1, 0]

def is_world_master(args):
    """
    This will be True only in one process, even in distributed mode,
    even when training on multiple machines.
    """
    # non-distributed training; local_rank value is not passed into argparser from Pytorch, so it has default value -1.
    return args.local_rank == -1 or torch.distributed.get_rank() == 0  


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def flatten_list(listoflists: List[List]):
    return [x for x in itertools.chain.from_iterable(listoflists)]
    # itertools.chain(*iterables) make an iterator that returns elements from the first iterable until it is exhausted, then proceeds to the next iterable, 
    # until all of the iterables are exhausted. Used for treating consecutive sequences as a single sequence. Roughly equivalent to:
    # def chain(*iterables):
    #     # chain('ABC', 'DEF') --> A B C D E F
    #     for it in iterables:
    #         for element in it:
    #             yield element

def lmap(f: Callable, x: Iterable) -> List:
    # apply a callable on each item of an iterable, and return a list
    return list(map(f, x))

def chunks(lst, chunk_size):
    # Yield successive chunk_size-sized chunks from lst.
    # So the input list is output to list of lists, each inner list being like a batch of size chunk_size.
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)

def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()

def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result

def get_max_step_model_path(output_dir, checks=1):
    assert checks >= 1 and checks <= 3
    # glob.glob(pathname, *, recursive=False) Return a possibly-empty list of path names that match pathname, which must be a string containing a path specification. 
    # pathname can be either absolute (like /usr/src/Python-1.5/Makefile) or relative (like ../../Tools/*/*.gif), and can contain shell-style wildcards. 
    fn_model_list = glob.glob(os.path.join(output_dir, PREFIX_CHECKPOINT_DIR + "*/%s" % WEIGHTS_NAME))
    fn_optim_list = glob.glob(os.path.join(output_dir, PREFIX_CHECKPOINT_DIR + "*/%s" % OPTIMIZER_NAME)) if checks > 1 else fn_model_list
    fn_sched_list = glob.glob(os.path.join(output_dir, PREFIX_CHECKPOINT_DIR + "*/%s" % SCHEDULER_NAME)) if checks > 2 else fn_model_list
    if (not fn_model_list) or (not fn_optim_list) or (not fn_sched_list):
        return None
    intersect = set([int(os.path.dirname(fn).split('-')[-1].split(os.path.sep)[0]) for fn in fn_model_list]
                   ) & set([int(os.path.dirname(fn).split('-')[-1].split(os.path.sep)[0]) for fn in fn_optim_list]
                   ) & set([int(os.path.dirname(fn).split('-')[-1].split(os.path.sep)[0]) for fn in fn_sched_list])
    if intersect:
        max_step = max(intersect)
        for fn in fn_model_list:
            if str(max_step) in fn:
                return os.path.dirname(fn)                
    else:
        return None


def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer, skip_special_tokens=True, special_token_ids_to_remove=None, clean_up_tokenization_spaces=False) -> Callable[[EvalPrediction], Dict]:
    # task_name: one of [summarization, translation, ]
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        # EvalPrediction is a NamedTuple that contains predicted sequence and target sequence token_ids of all examples
        #   predictions: Union[np.ndarray, Tuple[np.ndarray]]
        #   label_ids: np.ndarray
        
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=skip_special_tokens)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=skip_special_tokens)

        # can also add additional logic to clean up the resulting strings here
        if clean_up_tokenization_spaces:
            pred_str = lmap(clean_up_spaces_tokenization, pred_str)
            label_str = lmap(clean_up_spaces_tokenization, label_str)

        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)

        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)  # round to 1 decimal digits.
        rouge.update({"gen_len": summ_len})  # average length of generated sequence in tokens (excluding paddings)
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
    return compute_metrics_fn


def ids_to_clean_text(tokenizer, token_ids, special_token_ids_to_remove=None, skip_special_tokens=False, clean_up_tokenization_spaces=False):
    if not isinstance(token_ids, list):
        token_ids = token_ids.tolist()
    if special_token_ids_to_remove:
        token_ids = remove_special_token_ids(token_ids, special_token_ids_to_remove)
    gen_text = tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)  
    if clean_up_tokenization_spaces:
        gen_text = lmap(clean_up_spaces_tokenization, gen_text)
    return lmap(str.strip, gen_text)

def remove_special_token_ids(ids, special_token_ids=None):
    if not special_token_ids:
        return ids
    new_ids = []
    for id in ids:
        if isinstance(id, List):
            new_ids.append(remove_special_token_ids(id, special_token_ids))
        else:
            if id in special_token_ids:
                continue
            new_ids.append(id)
    return new_ids


def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

    Args:
        out_string (:obj:`str`): The text to clean up.

    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")  # how about double quotes "? English doesn't use single quote in pair? 
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string


def clean_up_spaces_tokenization(text):
    text = re.sub(r" - ", "-", text)  # causing issues?
    
    text = re.sub(r" n't ", "n't ", text)
    text = re.sub(r" 's ", "'s ", text)
    text = re.sub(r" 'll ", "'ll ", text)
    text = re.sub(r" 're ", "'re ", text)
    text = re.sub(r" 've ", "'ve ", text)
    text = re.sub(r" 'd ", "'d ", text)
    text = re.sub(r" 'm ", "'m ", text)

    text = re.sub(r"\. Com", ".com", text)
    text = re.sub(r"\. com", ".com", text)

    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" !", "!", text)
    text = re.sub(r" ' ", "'", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r" :", ":", text)
    text = re.sub(r" ;", ";", text)
    text = re.sub(r"\$ ", "$", text)
    text = re.sub(r"# ", "#", text)
    text = re.sub(r" %", "%", text)
    text = re.sub(r"\( ", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r"< ", "<", text)
    text = re.sub(r" >", ">", text)
    text = re.sub(r"\[ ", "[", text)
    text = re.sub(r" ]", "]", text)

    text = re.sub(r"(\d) ([\"\'])", r"\1\2", text)
    text = re.sub(r"([\']) (\d)", r"\1\2", text)

    text = re.sub(r'" ([^"]+) "', r'"\1"', text)
    text = re.sub(r"' ([^']+) '", r"'\1'", text)

    text = re.sub(r"([sS]) ' ", r"\1' ", text)

    text = re.sub(r" ' ", r"'", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # change website to lower case
    # text = re.sub(r"(\w+)\.com", lambda x: x.group(1).lower() + '.com', text)
 
    return text    


from sacrebleu import corpus_bleu
# SacreBLEU knows about common WMT test sets, and handles downloading, processing, and tokenization for you.
# But you can also use it to score system outputs (means model outputs) with arbitrary references. The system output and references will all be tokenized internally.
# corpus_bleu(sys_stream: Union[str, Iterable[str]], ref_streams: Union[str, List[Iterable[str]]], **kwargs) -> BLEUScore:
#    Produces BLEU scores along with its sufficient statistics from a source against one or more references.
#    sys_stream: The system stream (one or a list of model outupts)
#    ref_streams: One or a list of one or more reference texts (each item of the list has the same length as the sys_stream)


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    # output_lns is a list of str, each str is a prediction
    # refs_lns is a list of str, each str is a reference/label
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}  # round to #digits=4


# BLEU uses a modified form of precision to compare a model output against multiple references. 
# The modification is that for each distinct n-gram in the model output, BLEU caps its count to the maximum count of that n-gram in any one of the references.
# These capped counts are then summed over all distinct n-grams in the model output. This sum is then divided by the total number ofn-grams in the model output.
# One problem with BLEU scores is that they tend to favor short model outputs, which can produce very high precision scores, even using modified precision.
# It has been pointed out that precision is usually twinned with recall to overcome this problem.
# The problem being that as there are multiple references, a bad model output could easily have an inflated recall, such as a model output which consisted of all the words in each of the references.

# To produce a single BLEU score for the whole corpus, the modified precision scores for all n-grams are combined using the weighted geometric mean, different weights for different n-grams,
# then multiplied by a brevity penalty to prevent very short candidates from receiving too high a score. 
# Let r be the total length of the reference corpus, and c the total length of the model outupt corpus. If c <= r, the brevity penalty applies, defined to be e^{(1-r/c)}. 
# In the case of multiple reference sentences, r is taken to be the sum of the lengths of the references whose lengths are closest to the lengths of the model output sentences. 

# The length which has the "highest correlation with monolingual human judgements" was found to be four. 
# The unigram scores are found to account for the adequacy of the translation, or how much information is retained. 
# The longer n-gram scores account for the fluency of the translation, or to what extent it reads like "good English". 

# BLEU's output is always a number between 0 and 1. Because there are more opportunities to match, adding additional reference translations will increase the BLEU score.

# !! ROUGE uses both precision and recall! BLEU is precision based.
# BLEU tried to uses recall, but since BLEU can have multiple references for a single model output, it is difficult to formulate recall over multiple references.
# So BLEU uses a brevity penalty to compensate for the possibility of high-precision model outputs (hypothesis) which are too short.

# The difference between the ROUGE-n precision and BLEU is that BLEU introduces a brevity penalty term, and also compute a combined score (by weighted geometric mean) for several size of n-grams match. 
# ROUGE reports scores (can be precision, recall, or F1) of each n (for n-gram) (also LCS) individually, as ROUGE-n (and ROUGE-L). 



from rouge_score import rouge_scorer, scoring

# rouge_score package is designed to replicate original paper perl results. 
# Lin, Chin-Yew. ROUGE: a Package for Automatic Evaluation of Summaries. In Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004), Barcelona, Spain, July 25 - 26, 2004.
# It implements:
#    ROUGE-N (N-gram) scoring
#    ROUGE-L (Longest Common Subsequence) scoring
#    Text normalization
#    Bootstrap resampling for confidence interval calculation
#    Optional Porter stemming to remove plurals and word suffixes such as (ing, ion, ment).
# Does not include stopword removal.
# In the ROUGE paper, two flavors of ROUGE are described:
#    sentence-level: Compute longest common subsequence (LCS) between two pieces of text. Newlines are ignored. This is called rougeL in this package.
#    summary-level: Newlines in the text are interpreted as sentence boundaries, and the LCS is computed between each pair of reference and candidate sentences, 
#                   and something called union-LCS is computed. This is called rougeLsum in this package. 
#                   This is the ROUGE-L reported in Get To The Point: Summarization with Pointer-Generator Networks, for example.


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict

def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.
    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).
    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys
    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()  #  BootstrapAggregator(confidence_interval=0.95, n_samples=1000)

    for pred, tgt in zip(pred_lns, tgt_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pass
            #pred = add_newline_to_end_of_each_sentence(pred)
            #tgt = add_newline_to_end_of_each_sentence(tgt)
        
        scores = scorer.score(tgt, pred)
        # score(target, prediction) calculates rouge scores between the target (a str representing one target text) and prediction (a str representing one predict text).
        # Returns a dict mapping each rouge type to a Score object: return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)
        #           class Score(collections.namedtuple("Score", ["precision", "recall", "fmeasure"]))  # Tuple containing precision, recall, and f-measure values.
        # For lcs
        #   precision = lcs_length / len(prediction_tokens)
        #   recall = lcs_length / len(target_tokens)
        # For ngrams
        #   precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
        #   recall = intersection_ngrams_count / max(target_ngrams_count, 1)
        # And for every type of score, same: fmeasure = scoring.fmeasure(precision, recall)  
        # return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        aggregator.add_scores(scores)
        # scores is a Dict mapping score_type strings to scoring.Score object which is a namedtuple containing precision, recall, and f-measure values.
        # add_scores(self, scores):
        #    for score_type, score in six.iteritems(scores):
        #        self._scores[score_type].append(score)


    if bootstrap_aggregation:
        result = aggregator.aggregate()
        # aggregate() Aggregates scores previously added using add_scores. Returns: A dict mapping score_type to AggregateScore objects.
        # For example, result is {"rouge1": AggregateScore, "rouge2": AggregateScore, "rougeL": AggregateScore, "rougeLsum": AggregateScore}
        # class AggregateScore(collections.namedtuple("AggregateScore", ["low", "mid", "high"]))   # Tuple containing confidence intervals for scores.
        # low (row 0), mid (row 1) and high (row 2). mid is always the mean, while low and high depends on self._confidence_interval

        if return_precision_and_recall:
            # return a dict of of all requested rouge score types (i.e., "rouge1", "rouge2", "rougeL", "rougeLsum"), 
            # each value is a dict of mean score of "precision", "recall", "fmeasure" 
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            # return a dict of all requested rouge score types (i.e., "rouge1", "rouge2", "rougeL", "rougeLsum") and their mean scores of fmeasure
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)



def freeze_params(model: nn.Module):
    # Need to do this recursively?
    for par in model.parameters(): 
        par.requires_grad = False

def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())

def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))

def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    # Remove columns that are populated exclusively by pad_token_id
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def report_length(dataset, bin_size=16):
    # feature ids sequence length histogram
    for k, v in example.items():
        if not k in length_counter:
            length_counter[k] = collections.defaultdict(int)
        if k in passthrough:
            length_counter[k][len(v)] += 1
        else:
            length_counter[k][v["attention_mask"].sum().item()] += 1

    for k, v in length_counter.items():
        logger.info("Sequence length distribution of {}: ".format(k))
        report_length(v, total_count=len(features))

    # histogram bin width
    max_len = max(length_counter.keys())
    a = 0  # bin left end point. bin is [a, a + bin_size - 1]
    tc = 0
    while a < max_len:
        cc = 0
        for i in range(bin_size):
            # i is in [0, bin_size - 1]
            cc += length_counter[a + i]

        tc += cc
        if cc > 0:
            logger.info("%d ~ %d = %d, %.2f%%" % (a, a + bin_size, cc, (tc * 100.0) / total_count))
        a += bin_size  


def timer(func):
    def f(*args, **kwargs):
        print(f"#### Begin {func.__name__}")
        start = time.time()
        rv = func(*args, **kwargs)
        stop = time.time()
        print(f"#### End {func.__name__}. Time used: {stop - start:.1f}s\n")
        return rv
    return f



def batch_list_to_batch_tensors_with_passthrough(batch, passthrough=None):
    # passthrough is a set of indices in batch that can't be tensor, such as texts.
    batch_tensors = []
    for i, x in enumerate(zip(*batch)):
        try:
            if isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            elif passthrough and i in passthrough:
                batch_tensors.append(list(x))
            else:
                batch_tensors.append(torch.tensor(x))
        except:
            print("!!!! Batch data error !!!!")
            print("column index: ", i)
            print("data: ", x, "\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise ValueError("Data in batch have issues.")
    return batch_tensors



class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_file,
        column_names,
        max_source_length,
        max_target_length,
        split="train",
        header=False,
        sep='\t',
        n_obs=None,
        prefix="",
        cache_dir=None,
        **dataset_kwargs
    ):
        # column_names is a list of str, which is a subset of ['src', 'tgt', 'query', ...]
        super().__init__()

        assert split in ['train', 'validation', 'test'], "Valid split argument is one of ['train', 'validation', 'test']"
        skip_rows = 1 if header else 0
        
        data_dir = Path(data_file).parent
        cache_dir = data_dir if not cache_dir else cache_dir
        data_files = {split:data_file}

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        # Why do this for Bart?
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

        parse_options = pa.csv.ParseOptions( 
            delimiter=sep, 
            quote_char=False, 
            double_quote=False, 
            escape_char=False, 
            newlines_in_values=False, 
            ignore_empty_lines=False, 
        )

        dataset = load_dataset('csv', data_files=data_files, delimiter=sep, column_names=column_names, 
                    skip_rows=skip_rows, cache_dir=cache_dir)
        self.dataset = dataset[split]

        logger.info(f"dataset top view:\n{dataset}")
        logger.info(f'dataset cache files:\n{self.dataset.cache_files}')
        logger.info(f'data set first row:\n{self.dataset[0]}')

        # in trainer, data always have 'src' and 'tgt'. 
        def get_features(example):
            if self.prefix:
                update = {"src_texts": self.prefix + example["src_texts"]}
                return update

        self.dataset = self.dataset.map(get_features)        

        logger.info(f"processed dataset:\n{self.dataset}")
        logger.info(f"processed dataset first row:\n{self.dataset[0]}")

        if n_obs is not None and n_obs > 0:
            self.n_obs = n_obs
        else: 
            self.n_obs = len(self.dataset)
        logger.info(f"number of examples to generate: {self.n_obs}")

    def __len__(self):
        return self.n_obs

    def __getitem__(self, index):
        ex = self.dataset[index]
        return {**ex, "id": index}
        # what is "id" for? For sorting to get back original order of examples

    # why need this collate_fn if already use Seq2SeqDataCollator?
    # This collate_fn can be used in generation.
    # If for generation only, no need to compute loss, so no need to pass tgt_texts to prepare_seq2seq_batch().
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        # batch is a list of dict
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch] if "tgt_texts" in batch[0].keys() else None

        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            src_texts,
            #tgt_texts=tgt_texts,
            max_length=self.max_source_length,
            #max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        # This method calls tokenizer.__call__() to build a BatchEncoding, which derives from UserDict. 
        # some default args: padding="longest", truncation=True,
        # For classes derived from UserDict, an instance’s contents are kept in a regular dict, which is accessible via the 'data' attribute of UserDict instances.

        batch_encoding["src_texts"] = src_texts
        if tgt_texts:
            batch_encoding["tgt_texts"] = tgt_texts
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])  # for getting original order back after sorting examples
        return batch_encoding


def read_trainer_state(state_file: str):
    with open(state_file) as f:
        state = json.load(f)

    train_losses = collections.defaultdict(list)
    eval_losses = collections.defaultdict(list)

    for event in state['log_history']:
        if "loss" in event.keys():
            train_losses["step"].append(event['step'])
            train_losses["train_loss"].append(event['loss'])
        elif "eval_loss" in event.keys():
            eval_losses["step"].append(event['step'])
            eval_losses["eval_loss"].append(event['eval_loss'])
    
    return train_losses, eval_losses   

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    
    # ax.plot(eval_losses['step'], eval_losses['eval_loss'], c='r', marker="v", ls='--', label='eval_loss')
    # ax.plot(train_losses['step'], train_losses['train_loss'], c='b', ls='--', label='train_loss', fillstyle='none')

    # plt.xlabel('step')
    # plt.ylabel('loss')
    # plt.legend(loc=1)
    # plt.show()


######### My utils end #################
########################################################################################################


