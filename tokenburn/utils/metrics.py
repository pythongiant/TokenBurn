import math
import numpy as np

def compute_perplexity(logprobs):
    """
    Compute perplexity from logprobs.
    Formula: PPL = exp(- (1/N) * sum(log p_i))

    Parameters:
    - logprobs: list or array of log probabilities (log p_i) for each generated token.

    Returns:
    - Perplexity (float): A measure of how "confident" the model is.
      - Low perplexity (close to 1) → model is confident.
      - High perplexity → model is uncertain, more likely to hallucinate.
    """
    if len(logprobs) == 0:
        return float("inf")   # No tokens → undefined perplexity, return infinity
    avg_logprob = np.mean(logprobs)   # Average log probability across tokens
    return math.exp(-avg_logprob)     # Convert back to perplexity score


def compute_hallucination_risk(logprobs, threshold=0.2, entropy_thresh=2.5, var_thresh=0.5):
    """
    Compute hallucination risk using entropy and variance.

    Parameters:
    - logprobs: list/array of log probabilities (log p_i) for generated tokens.
    - threshold: baseline threshold (currently unused directly, but could be applied for cutoff logic).
    - entropy_thresh: maximum acceptable average entropy before marking as high risk.
        - Entropy measures uncertainty; higher entropy = model is more "unsure".
    - var_thresh: maximum acceptable variance in logprobs before marking as high risk.
        - Variance captures inconsistency in confidence across tokens.

    Returns:
    - Dictionary with:
        - risk: "LOW", "MEDIUM", or "HIGH" hallucination risk.
        - avg_entropy: average uncertainty across tokens.
        - variance: variance in token logprobs.
    """
    if len(logprobs) == 0:
        return {"risk": "UNKNOWN", "avg_entropy": None, "variance": None}

    # Convert log probabilities back to probabilities
    probs = np.exp(logprobs)

    # Normalize if it's a probability distribution over multiple tokens
    if probs.ndim > 1:
        probs = probs / probs.sum(axis=-1, keepdims=True)

    # Entropy calculation for each token probability distribution
    entropies = [-np.sum(p * np.log2(p + 1e-12)) for p in probs]
    avg_entropy = np.mean(entropies) if entropies else 0.0

    # Variance across token-level logprobs
    var = np.var(logprobs)

    # Assign risk level based on entropy + variance thresholds
    risk = "LOW"
    if avg_entropy > entropy_thresh or var > var_thresh:
        risk = "HIGH"
    elif avg_entropy > entropy_thresh * 0.7:
        risk = "MEDIUM"

    return {"risk": risk, "avg_entropy": avg_entropy, "variance": var}


def compute_optimal_cot_length(n, epsilon=0.1, H_cot=3, alpha=5, B0_minus_Bopt=6):
    """
    Compute the optimal Chain-of-Thought (CoT) length based on scaling laws.

    Formula:
    k* ≈ sqrt(alpha * n / (H_cot * (B0 - Bopt))) * log2(1/epsilon)

    Parameters:
    - n: number of available training samples or context size.
    - epsilon: target error tolerance (smaller epsilon = higher accuracy required).
    - H_cot: entropy cost of reasoning steps (larger H_cot = each reasoning step adds more "noise").
    - alpha: scaling constant controlling efficiency of reasoning vs. dataset size.
    - B0_minus_Bopt: performance gap between baseline and optimal reasoning.

    Returns:
    - Optimal chain-of-thought length (integer).
    """
    if n <= 0 or epsilon <= 0 or H_cot <= 0 or B0_minus_Bopt <= 0:
        return 0

    # Scaling law to compute best CoT length
    k = math.sqrt(alpha * n / (H_cot * (B0_minus_Bopt))) * math.log2(1 / epsilon)
    return int(round(k))
