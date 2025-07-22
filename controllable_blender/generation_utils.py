import math
import os
import json
from typing import List, Set, Dict, Union, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer

def cefr_to_int(cefr: str) -> int:
    mapping = {
        "A1": 0,
        "A2": 1,
        "B1": 2,
        "B2": 3,
        "C1": 4,
        "C2": 5,
    }
    clean_cefr = cefr.upper().strip()
    assert clean_cefr in mapping, f"CEFR must be one of {list(mapping.keys())}, not {cefr}"

    return mapping[clean_cefr]


def load_wordlist(path: str) -> List[str]:
    """
    Load a list of words from a text file containing one word per line
    """
    vocab = []

    if not path:
        return vocab

    assert os.path.isfile(path)

    with open(path, 'r', encoding="utf-8") as vocab_file:
        for row in vocab_file:
            token = row.strip()
            vocab.append(token)

    return vocab


class Wordlist():
    def __init__(self, vocab: List[str], tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

        # Identify token ID sequences that are allowed words
        # Identify allowed continuations of sequences
        self.allowed_token_seqs: Set[str] = set()
        self.allowed_continuations: Dict[str, List[int]] = {}

        for word in vocab:
            for word_variant in self._get_word_variants(word):
                token_ids = tokenizer.encode(word_variant, add_special_tokens=False)
                if not token_ids:
                    continue

                self.allowed_token_seqs.add(repr(token_ids))

                for i in range(1, len(token_ids)):
                    prefix = repr(token_ids[:i])  
                    next_token = token_ids[i]    # List represented as string for lookup
                    if prefix not in self.allowed_continuations:
                        self.allowed_continuations[prefix] = []
                    self.allowed_continuations[prefix].append(next_token)

    def get_allowed_ids(self, token_ids: List[int]) -> List[int]:
        """
        adapts parlai-based _get_continuation_ids function
        """
        last_word = self._get_last_word(token_ids)
        prefix_str = repr(last_word)
        continuation_ids = self.allowed_continuations.get(prefix_str, [])

        if self._is_word(last_word) or not last_word:
          continuation_ids += self._get_starting_tokens()

        return list(set(continuation_ids))


    def _is_word(self, token_ids: List[int]) -> bool:
        """
        For a given sequence of token IDs, determine whether that sequence is a complete word
        """
        return repr(token_ids) in self.allowed_token_seqs


    def _get_last_word(self, token_ids: List[int]) -> List[int]:
        """
        Get the sequence of token IDs after the last word boundary.
        Assumes that a word boundary is denoted by punctuation or whitespace (Ġ).
        """

        if not token_ids:
          return []

        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        for i in range(len(token_ids) -1, -1, -1):
            if tokens[i].startswith("Ġ") or tokens[i].startswith("_"):
                return token_ids[i:]
        return token_ids


    def _get_starting_tokens(self) -> List[int]:
        """ 
        return token IDs that can start a word (e.g. Ġ or_ initial)
        """
        vocab = self.tokenizer.get_vocab()
        starting_tokens = [idx for tok, idx in vocab.items() if tok.startswith("Ġ") or tok.startswith("_") or tok.isalpha()]
        return starting_tokens


    def _get_word_variants(self, word: str) -> Set[str]:
        return {word, word.lower(), word.capitalize()}



class Reranker():
    def __init__(self,
                 cefr: int,
                 model: str,
                 tokenizer: str = "distilroberta-base",
                 device: Optional[str] = "cuda",
                 text_truncate: int = 128,
                 exempt_tokens: Union[str, List[int]] = "all",
                 penalty_stddev: int = 2,
                 vocab_size: int = 8008,
                 word_filter: Optional[List[str]] = None):

        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        self.model = RobertaForSequenceClassification.from_pretrained(model)
        self.model.to(device)
        self.device = device

        self.target_cefr = cefr
        self.text_truncate = text_truncate
        self.word_filter = word_filter

        cefr_filepath = os.path.join(os.path.dirname(__file__), 'tokens_by_cefr.json')
        with open(cefr_filepath, 'r') as cefr_file:
            token_cefrs = json.load(cefr_file)

        if exempt_tokens == "all" or penalty_stddev < 0:      # No penalties
            self.token_penalties = torch.tensor([[1] * vocab_size])
        else:
            # calculate penalties per CEFR level difference (0 = same CEFR)
            normal_dist = torch.distributions.normal.Normal(0, penalty_stddev)
            cefr_penalties = [math.exp(normal_dist.log_prob(torch.tensor(i))) for i in range(6)]

            token_penalties = []
            for i in range(vocab_size):
                if i in exempt_tokens:
                    token_penalties.append(cefr_penalties[0])

                elif str(i) in token_cefrs:
                    token_str, token_cefr = token_cefrs[str(i)]
                    penalty = cefr_penalties[int(token_cefr - self.target_cefr)]

                    if token_cefr <= self.target_cefr or not token_str.isalpha():         # ignore lower CEFR levels and punctuation/special tokens
                        penalty = cefr_penalties[0]

                    token_penalties.append(penalty)

                else:       # Assume highest CEFR level if we don't have an assigned CEFR level
                    token_penalties.append(cefr_penalties[int(5 - self.target_cefr)])

            self.token_penalties = torch.tensor([token_penalties])

    def get_complexity_scores(self, hyps: List[str]) -> np.ndarray:
        model_inputs = self.tokenizer(hyps,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.text_truncate,
                                      return_tensors='pt',
                                      return_token_type_ids=True,
                                      return_attention_mask=True)

        model_output = self.model(input_ids=model_inputs["input_ids"].to(self.device),
                                  attention_mask=model_inputs["attention_mask"].to(self.device),
                                  token_type_ids=model_inputs["token_type_ids"].to(self.device))

        complexity_scores = model_output.logits.cpu().numpy().flatten()
        complexity_diffs = 5 - np.absolute(complexity_scores - self.target_cefr)      # reversed so that higher score = better

        return complexity_diffs

