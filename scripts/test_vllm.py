try:
    from vllm import LLM, SamplingParams
    USE_VLLM = True
except ImportError:
    USE_VLLM = False

if USE_VLLM:
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    params = SamplingParams(temperature=0.7)
    outputs = llm.generate("Why is the sky blue?", sampling_params=params)
    print(outputs[0].outputs[0].text)
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("⚠️ Using Transformers fallback (CPU-only)")
    model_id = "sshleifer/tiny-gpt2"  # very small model for local dev

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    inputs = tokenizer("Why is the sky blue?", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

