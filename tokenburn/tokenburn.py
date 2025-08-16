# tokenburn/tokenburn.py
from .utils.metrics import (
    compute_perplexity,
    compute_hallucination_risk,
    compute_optimal_cot_length,
    info_theoretic_score
)

class TokenBurn:
    def __init__(self, url, model, api_key):
        self.url = url
        self.model = model
        self.api_key = api_key

    def get_logprobs(self, messages, max_tokens=50, top_logprobs=5):
        """
        Call your HF endpoint, return logprobs for analysis.
        """
        import requests, json
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": top_logprobs
        }
        resp = requests.post(self.url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        logprobs = [t["logprob"] for t in data["choices"][0]["logprobs"]["content"]]
        return logprobs

    def perplexity(self, logprobs):
        return compute_perplexity(logprobs)

    def hallucination_risk(self, logprobs):
        return compute_hallucination_risk(logprobs)

    def find_optimal_cot_length(self, n, epsilon=0.1):
        return compute_optimal_cot_length(n, epsilon)

    def info_theoretic_score(self, token_logprob_seqs, token_index_seqs=None, reference_token_probs_seqs=None):
        return info_theoretic_score(token_logprob_seqs, token_index_seqs, reference_token_probs_seqs)
