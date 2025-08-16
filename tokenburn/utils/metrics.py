import math
import numpy as np
import math
import numpy as np
from collections import Counter

def compute_perplexity(logprobs):
    """
    Compute perplexity given a list of token logprobs.
    Perplexity = exp(- average log probability)
    """
    if not logprobs:
        return float("inf")
    avg_logprob = sum(logprobs) / len(logprobs)
    return math.exp(-avg_logprob)


def compute_hallucination_risk(logprobs, threshold=-1.5):
    """
    Estimate hallucination risk from logprobs.
    If average logprob is below threshold, risk is high.
    Returns value in [0, 1].
    """
    if not logprobs:
        return 1.0
    avg_logprob = sum(logprobs) / len(logprobs)
    risk = 1 / (1 + math.exp(avg_logprob - threshold))
    return risk


def compute_optimal_cot_length(n, epsilon=0.9):
    """
    Finds optimal chain-of-thought (CoT) length.
    Currently: epsilon * n (fraction of available tokens).
    """
    return max(1, int(epsilon * n))


def compute_entropy(logprobs):
    """
    Shannon entropy of token distribution.
    Input: list of dicts or tuples with {token: prob}
    For simplicity, assumes logprobs are token logprobs.
    """
    probs = [math.exp(lp) for lp in logprobs]
    Z = sum(probs)
    probs = [p / Z for p in probs]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def compute_topk_mass(logprobs, k=5):
    """
    Compute probability mass of top-k tokens.
    Input: list of logprobs (already sorted is fine).
    """
    probs = sorted([math.exp(lp) for lp in logprobs], reverse=True)
    return sum(probs[:k]) / sum(probs)


def compute_answer_diversity(responses):
    """
    Measures diversity across multiple model responses.
    Diversity = 1 - (frequency of most common answer / total answers).
    """
    if not responses:
        return 0.0
    counts = Counter(responses)
    most_common = counts.most_common(1)[0][1]
    return 1 - (most_common / len(responses))


def compute_calibration(logprobs, correct_token):
    """
    Calibration: checks if model assigns high probability to the correct token.
    Returns probability assigned to correct token.
    """
    token_probs = {i: math.exp(lp) for i, lp in enumerate(logprobs)}
    Z = sum(token_probs.values())
    token_probs = {i: p / Z for i, p in token_probs.items()}
    return token_probs.get(correct_token, 0.0)


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



EPS = 1e-12

def _softmax_from_logprobs(logps):
    """logps: list/np.array of log probabilities for whole vocab (or truncated top-k).
       returns normalized probs"""
    logps = np.array(logps, dtype=float)
    # numeric stable softmax
    m = np.max(logps)
    ex = np.exp(logps - m)
    probs = ex / (np.sum(ex) + EPS)
    return probs

def avg_cross_entropy_from_sequences(token_logprob_seqs, token_index_seqs):
    """
    token_logprob_seqs: list of arrays; each array contains log probs for the vocab at that step.
    token_index_seqs: list of indices of the observed tokens at each step.
    Returns average cross-entropy (nats). If logprobs are base-e, outputs nats.
    """
    if not token_logprob_seqs:
        return float('inf')
    n = len(token_logprob_seqs)
    total = 0.0
    for logps, t in zip(token_logprob_seqs, token_index_seqs):
        probs = _softmax_from_logprobs(logps)
        pt = probs[t] if t < len(probs) else EPS
        total += -math.log(pt + EPS)
    return total / n

def avg_entropy_from_sequences(token_logprob_seqs):
    if not token_logprob_seqs:
        return 0.0
    ent = 0.0
    for logps in token_logprob_seqs:
        probs = _softmax_from_logprobs(logps)
        ent += -np.sum(probs * np.log(probs + EPS))
    return ent / len(token_logprob_seqs)

def avg_kl_from_sequences(token_logprob_seqs, reference_token_probs_seqs):
    """
    reference_token_probs_seqs: list of reference distributions q (same shape as model)
    """
    if not token_logprob_seqs:
        return 0.0
    total = 0.0
    for logps, q in zip(token_logprob_seqs, reference_token_probs_seqs):
        p = _softmax_from_logprobs(logps)
        q = np.array(q, dtype=float)
        q = q / (q.sum() + EPS)
        # KL(q || p) = sum q * log(q/p)
        total += np.sum(q * (np.log(q + EPS) - np.log(p + EPS)))
    return total / len(token_logprob_seqs)

def info_theoretic_score(token_logprob_seqs,
                         token_index_seqs=None,
                         reference_token_probs_seqs=None,
                         vocab_size=50000,
                         weights=None,
                         caps=None):
    """
    Returns dict with components and final trust score in [0,1].
    - token_logprob_seqs: list of arrays of logprobs (per-step)
    - token_index_seqs: indices of observed tokens (to compute surprisal). Optional.
    - reference_token_probs_seqs: optional list of reference q distributions (for KL).
    """
    if weights is None:
        # if no reference KL, split mass across S and E more
        weights = dict(s=0.45, e=0.35, k=0.15, c=0.05)

    # caps for normalization
    if caps is None:
        caps = dict(S_max=math.log(vocab_size + EPS), K_max=math.log(vocab_size + EPS))

    # compute components
    S = 0.0
    if token_index_seqs is not None:
        S = avg_cross_entropy_from_sequences(token_logprob_seqs, token_index_seqs)  # nats
    else:
        # fallback: use average surprisal estimated by using highest-prob token index (pessimistic)
        S = avg_entropy_from_sequences(token_logprob_seqs)

    E = avg_entropy_from_sequences(token_logprob_seqs)  # nats
    K = 0.0
    if reference_token_probs_seqs is not None:
        K = avg_kl_from_sequences(token_logprob_seqs, reference_token_probs_seqs)
    # calibration penalty: average (1 - p_true)
    C = 0.0
    if token_index_seqs is not None:
        total_pen = 0.0
        for logps, t in zip(token_logprob_seqs, token_index_seqs):
            p = _softmax_from_logprobs(logps)
            total_pen += (1.0 - float(p[t] if t < len(p) else 0.0))
        C = total_pen / len(token_logprob_seqs)

    # Normalize to [0,1]
    norm_s = min(S / (caps.get("S_max") + EPS), 1.0)
    norm_e = min(E / (math.log(vocab_size + EPS) + EPS), 1.0)
    norm_k = 0.0
    if reference_token_probs_seqs is not None:
        norm_k = min(K / (caps.get("K_max") + EPS), 1.0)
    norm_c = min(C, 1.0)

    # If user didn't provide KL references, redistribute weight proportionally
    w = dict(s=weights.get("s",0.0), e=weights.get("e",0.0),
             k=weights.get("k",0.0), c=weights.get("c",0.0))
    if reference_token_probs_seqs is None and w["k"] > 0:
        # shift k into s and e
        shift = w["k"]
        w["k"] = 0.0
        w["s"] += 0.6 * shift
        w["e"] += 0.4 * shift

    # ensure normalization
    total_w = w["s"] + w["e"] + w["k"] + w["c"]
    if total_w <= 0:
        total_w = 1.0
        w = dict(s=0.5, e=0.5, k=0.0, c=0.0)
    for kkey in w:
        w[kkey] /= total_w

    risk = (w["s"] * norm_s +
            w["e"] * norm_e +
            w["k"] * norm_k +
            w["c"] * norm_c)

    trust = 1.0 - risk
    return dict(
        S=S, E=E, K=K, C=C,
        norm_s=norm_s, norm_e=norm_e, norm_k=norm_k, norm_c=norm_c,
        weights=w,
        risk=risk, trust=trust
    )