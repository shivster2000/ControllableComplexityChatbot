import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from generation_utils import Wordlist, Reranker

class ControllableDialoGPT:
    def __init__(self, config):
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history = None
        self.reranker = config.get("reranker", None)
        self.wordlist = config.get("wordlist", None)
        self.interactive_mode = config.get("interactive_mode", False)

    def set_interactive_mode(self, mode: bool):
        self.interactive_mode = mode

    def generate_response(self, user_input):
        input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt").to(self.device)
        full_input = torch.cat([self.history, input_ids], dim=-1) if self.history is not None else input_ids

        outputs = self.model.generate(
            full_input,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=20 if self.reranker else 1
        )

        candidates = [
            self.tokenizer.decode(out[full_input.shape[-1]:], skip_special_tokens=True)
            for out in outputs
        ]
        
        if self.wordlist:
            candidates = [c for c in candidates if self._passes_vocab_filter(c)]

        if not candidates:
            return "[No valid response]"

        if self.reranker:
            candidates = [self.tokenizer.decode(out[full_input.shape[-1]:], skip_special_tokens=True) for out in outputs]
            best_response = self.reranker.rank(candidates)
        else:
            best_response = self.tokenizer.decode(outputs[:, full_input.shape[-1]:][0], skip_special_tokens=True)

        self.history = torch.cat([full_input, self.tokenizer.encode(best_response + self.tokenizer.eos_token, return_tensors="pt").to(self.device)], dim=-1)
        return best_response

    def _passes_vocab_filter(self, text: str) -> bool:

        tokens = text.strip().split()
        return all(word.lower().strip(".,!?") in self.wordlist.allowed_token_seqs for word in tokens)
