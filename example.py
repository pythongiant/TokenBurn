from tokenburn.tokenburn import TokenBurn

tb = TokenBurn(url="https://router.huggingface.co/v1/chat/completions",model="openai/gpt-oss-120b:cerebras",api_key="")

# Example prompt
logprobs = tb.get_logprobs(messages= [
        {"role": "user", "content": 
"Who won the Nobel Prize in Physics in 2029"
}
    ], max_tokens=50,top_logprobs=5)

print("Perplexity:", tb.perplexity(logprobs))
print("Hallucination Risk:", tb.hallucination_risk(logprobs))
print("Optimal CoT Length:", tb.find_optimal_cot_length(n=len(logprobs), epsilon=0.9))

# Example prompt
logprobs = tb.get_logprobs(messages= [
        {"role": "user", "content": """
Respond in plain text, one word answers. 
What is the capital of France?
"""}
    ], max_tokens=50,top_logprobs=5)

print("Perplexity:", tb.perplexity(logprobs))
print("Hallucination Risk:", tb.hallucination_risk(logprobs))
print("Optimal CoT Length:", tb.find_optimal_cot_length(n=len(logprobs), epsilon=0.9))
