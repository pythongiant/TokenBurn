import requests
from .utils.metrics import compute_perplexity, compute_hallucination_risk, compute_optimal_cot_length
import json

class TokenBurn:
    def __init__(self, model="", url="",api_key="",
                 cot_length=0, max_retries=0, responses=True,
                 threshold=0, find_optimal_threshold=True):
        """
        TokenBurn Main class for OptiLLM API.
        """
        self.url = url
        self.cot_length = cot_length
        self.model_name = model
        self.max_retries = max_retries
        self.threshold = threshold
        self.find_optimal_threshold = find_optimal_threshold
        self.responses = responses
        self.api_key = api_key

    def get_logprobs(self, messages=[], top_logprobs=0, max_tokens=50, stream=False):
        """
        Query inference server and return token logprobs.
        """
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream,
                "logprobs": True,
                "top_logprobs": top_logprobs
            }
        )

        response_data = response.json()  # already a dict
        logprobs = []

        # Extract logprobs from Hugging Face/OpenAI-like response
        try:
            choices = response_data.get("choices", [])
            if choices:
                logprob_content = choices[0].get("logprobs", {}).get("content", [])
                for token_info in logprob_content:
                    if "logprob" in token_info:
                        logprobs.append(token_info["logprob"])
        except Exception as e:
            print("Error extracting logprobs:", e)
            print("Full response:", response_data)

        return logprobs
        
    def perplexity(self, logprobs):
        """Compute perplexity from logprobs."""
        return compute_perplexity(logprobs)

    def hallucination_risk(self, logprobs):
        """Assess hallucination risk from logprobs."""
        return compute_hallucination_risk(logprobs, self.threshold)

    def find_optimal_cot_length(self, n, epsilon=0.1, H_cot=3, alpha=5, B0_minus_Bopt=6):
        """Compute optimal chain-of-thought length based on scaling law."""
        return compute_optimal_cot_length(n, epsilon, H_cot, alpha, B0_minus_Bopt)
