# 🔥 TokenBurn

## 📚 References & Inspiration

This library was inspired by recent research on log-probabilities and Bayesian interpretations of LLMs:

* **“LLMs are Bayesian, in Expectation, not in Realization”**
  [arXiv:2507.11768](https://arxiv.org/pdf/2507.11768)

These ideas motivated the **hallucination risk scoring**, **information-theoretic metrics**, and **optimal CoT length scaling** in TokenBurn.

---

**TokenBurn** is a lightweight Python library for analyzing the confidence of Large Language Models (LLMs).
It helps you detect **hallucinations**, compute **perplexity**, and find the **optimal chain-of-thought (CoT) length** using scaling laws + information-theoretic metrics.

It works with any Hugging Face / OpenAI-compatible inference server that supports logprobs.

---

## 🚀 Features

* ✅ Extract **logprobs** from LLM outputs
* ✅ Compute **perplexity** (model confidence)
* ✅ Detect **hallucination risk** using entropy + variance
* ✅ Estimate **optimal CoT length** with scaling law
* ✅ Compute **information-theoretic metrics** (Entropy, KL Divergence, Mutual Information)
* ✅ Compatible with Hugging Face Inference API / OpenAI-like endpoints

---

## 📦 Installation

```bash
git clone https://github.com/yourname/tokenburn.git
cd tokenburn
pip install -r requirements.txt
```

---

## ⚡ Quickstart

```python
from tokenburn.tokenburn import TokenBurn

tb = TokenBurn(
    url="https://router.huggingface.co/v1/chat/completions",
    model="openai/gpt-oss-120b:cerebras",
    api_key="hf_xxxxx"
)

logprobs = tb.get_logprobs(messages=[
    {"role": "user", "content": "Who won the Nobel Prize in Physics in 2029?"}
], max_tokens=50, top_logprobs=5)

print("Perplexity:", tb.perplexity(logprobs))
print("Hallucination Risk:", tb.hallucination_risk(logprobs))
print("Optimal CoT Length:", tb.find_optimal_cot_length(n=len(logprobs), epsilon=0.9))
print("Entropy:", tb.entropy(logprobs))
print("KL Divergence vs. uniform:", tb.kl_divergence(logprobs, baseline="uniform"))
```

---

## 📊 Metrics Explained

### 🔹 Perplexity

* Confidence score from token logprobs:

$$
\text{PPL} = \exp\Bigg(-\frac{1}{N} \sum_{i=1}^N \log p(x_i)\Bigg)
$$

* **Low (≈1–20)** → confident prediction
* **High (>50)** → uncertain, likely hallucination

---

### 🔹 Hallucination Risk

* Combines **entropy** (spread of probability distribution) + **variance** (instability of logprobs)
* Risk levels: **LOW / MEDIUM / HIGH**

---

### 🔹 Optimal CoT Length

* Scaling law for reasoning length:

$$
k^* \approx \sqrt{\frac{\alpha n}{H_\text{cot} (B_0 - B_\text{opt})}} \cdot \log_2\Big(\frac{1}{\epsilon}\Big)
$$

* Predicts how many reasoning steps are optimal before diminishing returns

---

### 🔹 Information-Theoretic Metrics

* **Entropy (H)** – Uncertainty of the model:

$$
H(p) = -\sum_x p(x) \log p(x)
$$

* **KL Divergence**  – How much the model’s distribution deviates from a baseline:

$$
D_\text{KL}(p || q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

* **Mutual Information (I)** – Measures how much knowing the context reduces uncertainty about the next token:

$$
I(X;Y) = H(X) - H(X|Y)
$$

These metrics give a **principled, information-theoretic view** of model confidence and hallucination risk.

---

## 📂 Project Structure

```
tokenburn/
│── tokenburn.py         # Main TokenBurn class
│── utils/
│    └── metrics.py      # Perplexity, entropy, KL divergence, CoT length, etc.
```

---

## ✅ Example Use Cases

* Detect when your LLM is **guessing vs. confident**
* Use entropy & KL divergence for **early hallucination detection**
* Benchmark models on **information-theoretic efficiency**
* Tune reasoning dynamically with **optimal CoT scaling**

---

## 🛠 Roadmap

* [ ] Add visualization for perplexity & entropy
* [ ] Add mutual information trend plots
* [ ] Streaming logprobs support
* [ ] Benchmark suite with hallucination prompts

---

## 📜 License

MIT License – free to use and modify

---

