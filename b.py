from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello! Introduce yourself briefly in two sentences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

response = client.completions.create(
    model="/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    prompt=prompt,
    max_tokens=100,
    temperature=0.6,
    stop=["<|eot_id|>", "<|end_of_text|>"],
    frequency_penalty=0.5,
)

print("Model Response:\n")
print(response.choices[0].text.strip())
