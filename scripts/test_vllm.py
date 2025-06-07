import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["TMPDIR"] = "/tmp/mytmp"

from vllm import LLM, SamplingParams

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
params = SamplingParams(temperature=0.7)
outputs = llm.generate("Why is the sky blue?", sampling_params=params)
print(outputs[0].outputs[0].text)
