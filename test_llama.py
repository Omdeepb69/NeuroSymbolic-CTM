import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json

device = "cuda"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
llama_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

story = "Angela and her sister, Nancy, were looking through old photos. Their father, Samuel, walked in. Samuel's brother, Milton, was also there with his daughter, Arlene."

prompt = f"""Extract all explicitly stated family relationships from the story as a JSON list.
Format: [["Person1", "Person2", "relationship"], ...]
Use ONLY these words: father, mother, son, daughter, uncle, aunt, nephew, niece, brother, sister.

Story: {story}
JSON:"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = llama_model.generate(**inputs, max_new_tokens=200, do_sample=False)
ans = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"RAW LLAMA OUTPUT:\n{ans}")
