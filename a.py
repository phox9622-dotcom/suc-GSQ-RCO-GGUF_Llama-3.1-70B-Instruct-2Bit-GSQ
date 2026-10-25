from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

# صياغة النص بقالب Llama-3.1 الرسمي
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>how are you?.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

response = client.completions.create(
    model="/home/m/Desktop/GSQ-RCO-GGUF/Llama-3.1-70B-Instruct-2Bit-GSQ",
    prompt=prompt,
    max_tokens=100,
    temperature=0.6,
    stop=["<|eot_id|>", "<|end_of_text|>"],  # للتوقف فور إتمام الجواب
    frequency_penalty=0.5,  # لمنع التكرار
)

print("\nرد النموذج:\n")
print(response.choices[0].text.strip())
