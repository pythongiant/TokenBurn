
---

## 📚 References & Inspiration

This library was inspired by recent research on log-probabilities and Bayesian interpretations of LLMs:

* **“LLMs are Bayesian, in Expectation, not in Realization”**
  [arXiv:2507.11768](https://arxiv.org/pdf/2507.11768)

These ideas motivated the **hallucination risk scoring** and **optimal CoT length scaling** methods implemented in TokenBurn.

---
# 🔥 TokenBurn

**TokenBurn** is a lightweight Python library for analyzing the confidence of Large Language Models (LLMs).
It helps you detect **hallucinations**, compute **perplexity**, and find the **optimal chain-of-thought (CoT) length** using scaling laws.

It works with any Hugging Face / OpenAI-compatible inference server that supports logprobs.

---

## 🚀 Features

* ✅ Extract **logprobs** from LLM outputs
* ✅ Compute **perplexity** (model confidence)
* ✅ Detect **hallucination risk** using entropy + variance
* ✅ Estimate **optimal CoT length** with scaling law
* ✅ Compatible with Hugging Face Inference API / OpenAI-like endpoints

---

## 📦 Installation

Clone the repo and install requirements:

```bash
git clone https://github.com/yourname/tokenburn.git
cd tokenburn
pip install -r requirements.txt
```

Or install directly from local path:

```bash
pip install -e .
```

---

## ⚡ Quickstart
### Code:
```python
from tokenburn.tokenburn import TokenBurn

# Initialize TokenBurn with your model + endpoint
tb = TokenBurn(
    url="https://router.huggingface.co/v1/chat/completions",
    model="openai/gpt-oss-120b:cerebras",
    api_key="hf_xxxxx"   # Replace with your HF API key
)

# Example 1: A question that may cause hallucination
logprobs = tb.get_logprobs(messages=[
    {"role": "user", "content": "Who won the Nobel Prize in Physics in 2029?"}
], max_tokens=50, top_logprobs=5)

print("Perplexity:", tb.perplexity(logprobs))
print("Hallucination Risk:", tb.hallucination_risk(logprobs))
print("Optimal CoT Length:", tb.find_optimal_cot_length(n=len(logprobs), epsilon=0.9))

# Example 2: A well-known fact
logprobs = tb.get_logprobs(messages=[
    {"role": "user", "content": "What is the capital of France?"}
], max_tokens=50, top_logprobs=5)

print("Perplexity:", tb.perplexity(logprobs))
print("Hallucination Risk:", tb.hallucination_risk(logprobs))
print("Optimal CoT Length:", tb.find_optimal_cot_length(n=len(logprobs), epsilon=0.9))
```
### Output:
```text
Perplexity: 1.7654718506453648
Hallucination Risk: {'risk': 'HIGH', 'avg_entropy': 0.16010855028274482, 'variance': 1.310231613318954}
Optimal CoT Length: 1
Perplexity: 1.2230227331760852
Hallucination Risk: {'risk': 'LOW', 'avg_entropy': 0.096105875932574, 'variance': 0.23889273251666965}
Optimal CoT Length: 0
```
---

## 📊 Metrics Explained

### 🔹 Perplexity

* Measures how confident the model is in its predictions.
* **Low perplexity (≈1–20)** → high confidence, usually factual.
* **High perplexity (>50)** → uncertainty, higher chance of hallucination.

### 🔹 Hallucination Risk

* Based on:

  * **Entropy**: how spread out the model’s probability distribution is.
  * **Variance**: how much token confidence fluctuates.
* Returns **LOW / MEDIUM / HIGH**.

### 🔹 Optimal CoT Length

* Uses a scaling law:

  $$
  k^* ≈ \sqrt{\frac{αn}{H_{cot}(B_0 - B_{opt})}} \cdot \log_2\left(\frac{1}{\epsilon}\right)
  $$
* Helps decide how many reasoning steps the model should generate before stopping.

---

## 📂 Project Structure

```
tokenburn/
│── tokenburn.py         # Main TokenBurn class
│── utils/
│    └── metrics.py      # Perplexity, hallucination risk, CoT length functions
```

---

## ✅ Example Use Cases

* Detect when your LLM is **guessing vs. confident**.
* Auto-stop generation when hallucination risk is **HIGH**.
* Compare models by **perplexity stability**.
* Tune reasoning length dynamically with **optimal CoT scaling**.

---

## 🛠 Roadmap

* [ ] Add visualization for perplexity & entropy
* [ ] Support for streaming logprobs
* [ ] Prebuilt benchmarking suite with hallucination prompts

---

## 📜 License

MIT License – feel free to use and modify.

